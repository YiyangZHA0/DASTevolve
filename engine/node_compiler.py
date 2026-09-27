

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from astevolve.domain.dual_ast import FunctionalNodeSpec, StructuralNodeSpec
from astevolve.search.operator_registry import (
    OperatorConfigError,
    operator_supports_node_kind,
    require_operator,
)


EXECUTABLE_NODE_SET_VERSION = "astevolve.executable_ast_nodes.v1"


class ExecutableNodeCompileError(ValueError):
    pass


def _copied_fixed(
    value: Mapping[str, Mapping[int, str]],
) -> Dict[str, Dict[int, str]]:
    return {
        str(chain): {int(position): str(aa) for position, aa in residues.items()}
        for chain, residues in value.items()
    }


def _copied_masks(
    value: Mapping[str, Iterable[bool]],
) -> Dict[str, Tuple[bool, ...]]:
    return {
        str(chain): tuple(bool(item) for item in mask)
        for chain, mask in value.items()
    }


@dataclass(frozen=True)
class StructuralNodeContract:


    node_id: str
    declared_kind: str
    chain_id: str
    spans: Tuple[Tuple[int, int], ...]
    compiled_segment_name: str
    compiled_segment_kind: str
    selected_positions: Tuple[int, ...]
    legal_positions: Tuple[int, ...]
    invariant_kinds: Tuple[str, ...]
    allowed_operators: Tuple[str, ...]
    action_budgets: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "declared_kind": self.declared_kind,
            "selector": {
                "chain_id": self.chain_id,
                "spans": [list(span) for span in self.spans],
            },
            "compiled_segment_name": self.compiled_segment_name,
            "compiled_segment_kind": self.compiled_segment_kind,
            "selected_positions": list(self.selected_positions),
            "legal_positions": list(self.legal_positions),
            "invariant_kinds": list(self.invariant_kinds),
            "allowed_operators": list(self.allowed_operators),
            "action_budgets": {
                name: {"min": int(budget["min"]), "max": int(budget["max"])}
                for name, budget in sorted(self.action_budgets.items())
            },
        }


@dataclass(frozen=True)
class MeasurementIntent:


    functional_node_id: str
    kind: str
    state: str
    direction: str
    comparator: str
    evaluator_id: str
    term_name: str
    gate: Mapping[str, Any] | None
    missing_policy: str
    evidence_refs: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "functional_node_id": self.functional_node_id,
            "kind": self.kind,
            "state": self.state,
            "direction": self.direction,
            "comparator": self.comparator,
            "evaluator_binding": {
                "evaluator_id": self.evaluator_id,
                "term_name": self.term_name,
            },
            "gate": deepcopy(dict(self.gate)) if self.gate is not None else None,
            "missing_policy": self.missing_policy,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class ExecutableNodePlan:


    enabled: bool
    structural_nodes: Tuple[StructuralNodeContract, ...]
    measurement_intents: Tuple[MeasurementIntent, ...]
    effective_masks: Mapping[str, Tuple[bool, ...]]
    effective_fixed_residues: Mapping[str, Mapping[int, str]]
    node_policy_patches: Mapping[str, Mapping[str, Any]]
    schema_version: str = EXECUTABLE_NODE_SET_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": bool(self.enabled),
            "structural_nodes": [item.to_dict() for item in self.structural_nodes],
            "measurement_intents": [
                item.to_dict() for item in self.measurement_intents
            ],
            "effective_masks": {
                chain: list(mask) for chain, mask in sorted(self.effective_masks.items())
            },
            "effective_fixed_residues": {
                chain: {
                    str(int(position)): str(aa)
                    for position, aa in sorted(residues.items())
                }
                for chain, residues in sorted(
                    self.effective_fixed_residues.items()
                )
            },
            "node_policy_patches": {
                name: deepcopy(dict(patch))
                for name, patch in sorted(self.node_policy_patches.items())
            },
            "edge_execution": "unsupported_until_ast_01b",
        }


def _selector_positions(
    node: StructuralNodeSpec,
    template_sequences: Mapping[str, str],
) -> Tuple[int, ...]:
    selector = node.selector
    chain_id = str(selector.chain_id)
    if chain_id not in template_sequences:
        raise ExecutableNodeCompileError(
            f"structural node {node.node_id!r} selector references unknown chain {chain_id!r}"
        )
    length = len(template_sequences[chain_id])
    positions: List[int] = []
    for start, end in selector.spans:
        start, end = int(start), int(end)
        if start < 0 or end <= start or end > length:
            raise ExecutableNodeCompileError(
                f"structural node {node.node_id!r} selector span [{start}, {end}) "
                f"is outside chain {chain_id!r} length {length} bounds"
            )
        positions.extend(range(start, end))
    if not positions:
        raise ExecutableNodeCompileError(
            f"structural node {node.node_id!r} selector contains no residues"
        )
    if len(positions) != len(set(positions)):
        raise ExecutableNodeCompileError(
            f"structural node {node.node_id!r} selector contains overlapping spans"
        )
    return tuple(positions)


def _segment_indices(segment: Any) -> set[int]:
    if hasattr(segment, "indices"):
        return {int(position) for position in segment.indices()}
    positions: set[int] = set()
    for start, end in getattr(segment, "spans", ()) or ():
        positions.update(range(int(start), int(end)))
    return positions


def _resolve_segment(
    node: StructuralNodeSpec,
    positions: Sequence[int],
    compiled: Mapping[str, Any],
) -> Any:
    chain_id = str(node.selector.chain_id)
    position_set = set(int(position) for position in positions)
    matches = [
        segment
        for segment in compiled.get("segments", ()) or ()
        if str(getattr(segment, "chain_id", "")) == chain_id
        and position_set.issubset(_segment_indices(segment))
    ]
    if len(matches) != 1:
        raise ExecutableNodeCompileError(
            f"structural node {node.node_id!r} selector must resolve to exactly one "
            f"compiled segment; resolved {len(matches)}"
        )
    return matches[0]


def _action_contract(
    node: StructuralNodeSpec,
    segment: Any,
) -> Tuple[Tuple[str, ...], Dict[str, Dict[str, int]]]:
    operators: List[str] = []
    budgets: Dict[str, Dict[str, int]] = {}
    for action in node.allowed_actions:
        try:
            spec = require_operator(action.operator, mode="node")
        except OperatorConfigError as error:
            raise ExecutableNodeCompileError(
                f"structural node {node.node_id!r} allowed action operator is invalid: {error}"
            ) from error
        segment_kind = str(getattr(segment, "kind", "") or "")
        if not operator_supports_node_kind(spec.name, segment_kind):
            raise ExecutableNodeCompileError(
                f"operator {spec.name!r} is incompatible with compiled segment kind "
                f"{segment_kind!r} for structural node {node.node_id!r}"
            )
        if spec.name in budgets:
            raise ExecutableNodeCompileError(
                f"structural node {node.node_id!r} declares operator {spec.name!r} twice"
            )
        operators.append(spec.name)
        budgets[spec.name] = {
            "min": int(action.budget_min),
            "max": int(action.budget_max),
        }
    return tuple(operators), budgets


def _normalize_capabilities(
    value: Mapping[str, Iterable[str]] | None,
) -> Dict[str, set[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExecutableNodeCompileError("evaluator_capabilities must be a mapping")
    out: Dict[str, set[str]] = {}
    for evaluator_id, raw_terms in value.items():
        if isinstance(raw_terms, str):
            terms = {raw_terms}
        else:
            try:
                terms = {str(term) for term in raw_terms}
            except TypeError as error:
                raise ExecutableNodeCompileError(
                    f"evaluator capability {evaluator_id!r} must be an iterable of term names"
                ) from error
        if not terms or any(not term for term in terms):
            raise ExecutableNodeCompileError(
                f"evaluator capability {evaluator_id!r} has no valid term names"
            )
        out[str(evaluator_id)] = terms
    return out


def _measurement_intent(
    node: FunctionalNodeSpec,
    capabilities: Mapping[str, set[str]],
) -> MeasurementIntent:
    binding = node.evaluator_binding
    evaluator_id = str(binding.evaluator_id)
    term_name = str(binding.term_name)
    if evaluator_id not in capabilities or term_name not in capabilities[evaluator_id]:
        raise ExecutableNodeCompileError(
            f"functional node {node.node_id!r} evaluator capability is unavailable: "
            f"{evaluator_id!r}/{term_name!r}"
        )
    direction = str(node.direction)
    comparator = (
        "higher_is_better" if direction == "maximize" else "lower_is_better"
    )
    gate = None
    if str(node.kind) == "hard_constraint":
        if node.threshold is None:
            raise ExecutableNodeCompileError(
                f"hard_constraint functional node {node.node_id!r} requires a threshold"
            )
        gate = {
            "operator": ">=" if direction == "maximize" else "<=",
            "threshold": float(node.threshold),
        }
    return MeasurementIntent(
        functional_node_id=str(node.node_id),
        kind=str(node.kind),
        state=str(node.state),
        direction=direction,
        comparator=comparator,
        evaluator_id=evaluator_id,
        term_name=term_name,
        gate=gate,
        missing_policy=str(node.missing_policy),
        evidence_refs=tuple(str(item) for item in node.evidence_refs),
    )


def compile_executable_ast_nodes(
    raw: Mapping[str, Any] | None,
    *,
    compiled: Mapping[str, Any],
    template_sequences: Mapping[str, str],
    base_masks: Mapping[str, Iterable[bool]],
    base_fixed_residues: Mapping[str, Mapping[int, str]],
    evaluator_capabilities: Mapping[str, Iterable[str]] | None = None,
) -> ExecutableNodePlan:


    normalized_masks = _copied_masks(base_masks)
    normalized_fixed = _copied_fixed(base_fixed_residues)
    for chain, sequence in template_sequences.items():
        mask = normalized_masks.get(str(chain))
        if mask is None or len(mask) != len(sequence):
            raise ExecutableNodeCompileError(
                f"base mask length does not match template chain {chain!r}"
            )
        normalized_fixed.setdefault(str(chain), {})
    if raw is None:
        return ExecutableNodePlan(
            enabled=False,
            structural_nodes=(),
            measurement_intents=(),
            effective_masks=normalized_masks,
            effective_fixed_residues=normalized_fixed,
            node_policy_patches={},
        )
    if not isinstance(raw, Mapping):
        raise ExecutableNodeCompileError("executable_ast_nodes must be a mapping")
    allowed_top_level = {"schema_version", "structural_nodes", "functional_nodes"}
    unknown = sorted(set(str(key) for key in raw) - allowed_top_level)
    if unknown:
        raise ExecutableNodeCompileError(
            "executable node set contains unknown fields; mapping edges are unsupported "
            f"until AST-01B: {', '.join(unknown)}"
        )
    if raw.get("schema_version") != EXECUTABLE_NODE_SET_VERSION:
        raise ExecutableNodeCompileError(
            f"unsupported executable node set schema_version {raw.get('schema_version')!r}"
        )
    raw_structural = raw.get("structural_nodes")
    raw_functional = raw.get("functional_nodes")
    if not isinstance(raw_structural, list) or not isinstance(raw_functional, list):
        raise ExecutableNodeCompileError(
            "executable node set structural_nodes and functional_nodes must be lists"
        )
    structural_specs = tuple(
        StructuralNodeSpec.from_mapping(item) for item in raw_structural
    )
    functional_specs = tuple(
        FunctionalNodeSpec.from_mapping(item) for item in raw_functional
    )
    node_ids = [node.node_id for node in (*structural_specs, *functional_specs)]
    if len(node_ids) != len(set(node_ids)):
        raise ExecutableNodeCompileError(
            "executable structural and functional node IDs must be globally unique"
        )

    effective_mask_lists = {
        chain: [False] * len(mask) for chain, mask in normalized_masks.items()
    }

    for chain, residues in normalized_fixed.items():
        for position in residues:
            if chain in effective_mask_lists and 0 <= int(position) < len(
                effective_mask_lists[chain]
            ):
                effective_mask_lists[chain][int(position)] = False

    contracts: List[StructuralNodeContract] = []
    policy_patches: Dict[str, Dict[str, Any]] = {}
    claimed_positions: Dict[Tuple[str, int], str] = {}
    claimed_segments: Dict[str, str] = {}
    for node in structural_specs:
        positions = _selector_positions(node, template_sequences)
        segment = _resolve_segment(node, positions, compiled)
        segment_name = str(getattr(segment, "name", "") or "")
        if segment_name in claimed_segments:
            raise ExecutableNodeCompileError(
                f"AST-01A supports one structural contract per compiled segment; "
                f"{node.node_id!r} conflicts with {claimed_segments[segment_name]!r}"
            )
        claimed_segments[segment_name] = str(node.node_id)
        for position in positions:
            key = (str(node.selector.chain_id), int(position))
            if key in claimed_positions:
                raise ExecutableNodeCompileError(
                    f"structural node {node.node_id!r} selector overlaps node "
                    f"{claimed_positions[key]!r} at {key[0]}:{key[1]}"
                )
            claimed_positions[key] = str(node.node_id)
        declared_operators, budgets = _action_contract(node, segment)
        invariant_kinds = tuple(str(item.kind) for item in node.invariants)
        preserve_parent = "preserve_parent" in invariant_kinds
        frozen = str(node.kind) == "frozen" or preserve_parent
        if str(node.kind) == "frozen" and declared_operators:
            raise ExecutableNodeCompileError(
                f"frozen structural node {node.node_id!r} cannot declare allowed actions"
            )
        if str(node.kind) == "editable" and not preserve_parent and not declared_operators:
            raise ExecutableNodeCompileError(
                f"editable structural node {node.node_id!r} requires at least one allowed action"
            )
        chain_id = str(node.selector.chain_id)
        base_mask = normalized_masks[chain_id]
        legal_positions = tuple(
            position
            for position in positions
            if not frozen
            and bool(base_mask[position])
            and position not in normalized_fixed.get(chain_id, {})
        )
        if frozen:
            for position in positions:
                normalized_fixed[chain_id][position] = template_sequences[chain_id][
                    position
                ]
        for position in legal_positions:
            effective_mask_lists[chain_id][position] = True
        effective_operators = declared_operators if legal_positions else ()
        effective_budgets = (
            {name: budgets[name] for name in effective_operators}
            if effective_operators
            else {}
        )
        contract = StructuralNodeContract(
            node_id=str(node.node_id),
            declared_kind=str(node.kind),
            chain_id=chain_id,
            spans=tuple((int(start), int(end)) for start, end in node.selector.spans),
            compiled_segment_name=segment_name,
            compiled_segment_kind=str(getattr(segment, "kind", "") or ""),
            selected_positions=positions,
            legal_positions=legal_positions,
            invariant_kinds=invariant_kinds,
            allowed_operators=effective_operators,
            action_budgets=effective_budgets,
        )
        contracts.append(contract)
        if effective_operators:
            default_weight = 1.0 / float(len(effective_operators))
            policy_patches[segment_name] = {
                "enabled": True,
                "mutable": True,
                "mutation_ops": {
                    name: default_weight for name in effective_operators
                },
                "allowed_operators": list(effective_operators),
                "allowed_action_budgets": deepcopy(effective_budgets),
                "max_mutations_per_step": max(
                    budget["max"] for budget in effective_budgets.values()
                ),
                "ast_legal_positions": list(legal_positions),
                "executable_ast_node_id": str(node.node_id),
            }
        else:
            policy_patches[segment_name] = {
                "enabled": False,
                "mutable": False,
                "mutation_rate": 0.0,
                "max_mutations_per_step": 0,
                "edit_contract_action": "freeze_node",
                "ast_legal_positions": [],
                "executable_ast_node_id": str(node.node_id),
            }

    capabilities = _normalize_capabilities(evaluator_capabilities)
    intents = tuple(
        _measurement_intent(node, capabilities) for node in functional_specs
    )
    return ExecutableNodePlan(
        enabled=True,
        structural_nodes=tuple(contracts),
        measurement_intents=intents,
        effective_masks={
            chain: tuple(values) for chain, values in effective_mask_lists.items()
        },
        effective_fixed_residues=normalized_fixed,
        node_policy_patches=policy_patches,
    )


def merge_node_policy_patches(
    strategy: Mapping[str, Any],
    plan: ExecutableNodePlan,
    global_node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:


    updated = deepcopy(dict(strategy))
    if not plan.enabled:
        return updated
    policies = deepcopy(dict(updated.get("node_edit_policies") or {}))
    for segment_name, patch in plan.node_policy_patches.items():
        merged = deepcopy(dict(policies.get(segment_name) or {}))
        allowed = tuple(str(item) for item in patch.get("allowed_operators", ()) or ())
        existing_declares_ops = "mutation_ops" in merged
        existing_ops = merged.get("mutation_ops")
        merged.update(deepcopy(dict(patch)))
        if allowed and existing_declares_ops:
            if not isinstance(existing_ops, Mapping):
                raise ExecutableNodeCompileError(
                    f"node policy {segment_name!r} mutation_ops must be a mapping"
                )
            retained: Dict[str, float] = {}
            allowed_set = set(allowed)
            for raw_name, raw_weight in existing_ops.items():
                name = str(raw_name)
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError) as error:
                    raise ExecutableNodeCompileError(
                        f"node policy {segment_name!r} operator {name!r} has a "
                        "non-numeric weight"
                    ) from error
                if name in allowed_set and isfinite(weight) and weight > 0.0:
                    retained[name] = weight
            total = sum(retained.values())
            if total <= 0.0:
                requested = ", ".join(sorted(str(name) for name in existing_ops))
                raise ExecutableNodeCompileError(
                    f"node policy {segment_name!r} explicitly requests no operator "
                    f"allowed by its StructuralNode; requested [{requested}], "
                    f"allowed [{', '.join(allowed)}]"
                )
            merged["mutation_ops"] = {
                name: weight / total for name, weight in retained.items()
            }
        policies[str(segment_name)] = merged
    metadata = global_node_metadata if isinstance(global_node_metadata, Mapping) else {}
    contract_by_node = {
        contract.node_id: contract
        for contract in plan.structural_nodes
        if contract.declared_kind == "editable" and contract.legal_positions
    }
    for node_id, raw_metadata in metadata.items():
        contract = contract_by_node.get(str(node_id))
        if contract is None or not isinstance(raw_metadata, Mapping):
            continue
        segment_name = str(contract.compiled_segment_name)
        merged = deepcopy(dict(policies.get(segment_name) or {}))
        residue_policy = raw_metadata.get("residue_policy")
        if isinstance(residue_policy, Mapping):
            for field in (
                "favored_residues",
                "disfavored_residues",
                "position_residue_rules",
                "policy_weight",
            ):
                if field in residue_policy:
                    merged[field] = deepcopy(residue_policy[field])
        intent = str(raw_metadata.get("intent") or "").strip()
        if intent:
            merged["edit_intent"] = intent
        evidence_refs = raw_metadata.get("evidence_refs")
        if isinstance(evidence_refs, (list, tuple)):
            merged["ast_evidence_refs"] = [str(item) for item in evidence_refs]
        merged["ast_action_profile"] = str(
            raw_metadata.get("action_profile") or ""
        )
        policies[segment_name] = merged
    updated["node_edit_policies"] = policies
    active_nodes = [
        str(contract.compiled_segment_name)
        for contract in plan.structural_nodes
        if contract.declared_kind == "editable" and contract.legal_positions
    ]


    updated["semantic_required_nodes"] = list(active_nodes)
    updated["semantic_active_nodes"] = list(active_nodes)
    updated["_tree_policy_active"] = True
    updated["executable_ast_nodes"] = plan.to_dict()
    return updated


def apply_case_owned_residue_policy(
    strategy: Mapping[str, Any],
    plan: ExecutableNodePlan,
    raw_policy: Mapping[str, Any] | None,
) -> Dict[str, Any]:


    updated = deepcopy(dict(strategy))
    if raw_policy is None:
        return updated
    if not isinstance(raw_policy, Mapping):
        raise ExecutableNodeCompileError(
            "case_owned_residue_policy must be a mapping"
        )
    tiers = raw_policy.get("tiers")
    if not isinstance(tiers, Mapping) or not tiers:
        raise ExecutableNodeCompileError(
            "case_owned_residue_policy.tiers must be a non-empty mapping"
        )
    requested_tier = str(
        updated.get("case_owned_residue_policy_tier")
        or raw_policy.get("default_tier")
        or ""
    ).strip()
    if not requested_tier:
        raise ExecutableNodeCompileError(
            "case-owned residue policy requires an explicit or default tier"
        )
    if requested_tier not in tiers:
        raise ExecutableNodeCompileError(
            f"unknown case-owned residue policy tier {requested_tier!r}; "
            f"available tiers: {', '.join(sorted(map(str, tiers)))}"
        )
    tier = tiers[requested_tier]
    if not isinstance(tier, Mapping):
        raise ExecutableNodeCompileError(
            f"case-owned residue policy tier {requested_tier!r} must be a mapping"
        )
    tier_policies = tier.get("node_edit_policies")
    if not isinstance(tier_policies, Mapping) or not tier_policies:
        raise ExecutableNodeCompileError(
            f"case-owned residue policy tier {requested_tier!r} must declare "
            "node_edit_policies"
        )

    effective_policies = deepcopy(dict(updated.get("node_edit_policies") or {}))
    mutation_contract: Dict[str, Dict[int, list[str]]] = {}
    resolution: Dict[str, Any] = {
        "schema_version": "astevolve.case_owned_residue_policy_resolution.v1",
        "source_schema_version": str(raw_policy.get("schema_version") or ""),
        "active_tier": requested_tier,
        "authority": "case_controller",
        "active_segments": {},
    }
    canonical = "ACDEFGHIKLMNPQRSTVWY"

    for contract in plan.structural_nodes:
        if contract.declared_kind != "editable" or not contract.legal_positions:
            continue
        segment_name = str(contract.compiled_segment_name)
        declared_policy = tier_policies.get(segment_name)
        if not isinstance(declared_policy, Mapping):
            raise ExecutableNodeCompileError(
                f"tier {requested_tier!r} has no hard residue policy for "
                f"active segment {segment_name!r}"
            )
        declared_rules = declared_policy.get("position_residue_rules")
        if not isinstance(declared_rules, Mapping):
            raise ExecutableNodeCompileError(
                f"tier {requested_tier!r} segment {segment_name!r} must declare "
                "position_residue_rules"
            )

        merged_policy = deepcopy(dict(effective_policies.get(segment_name) or {}))
        merged_rules = deepcopy(
            dict(merged_policy.get("position_residue_rules") or {})
        )
        segment_resolution: Dict[str, list[str]] = {}
        for position in contract.legal_positions:
            raw_rule = declared_rules.get(
                str(int(position)),
                declared_rules.get(int(position)),
            )
            if not isinstance(raw_rule, Mapping):
                raise ExecutableNodeCompileError(
                    f"tier {requested_tier!r} segment {segment_name!r} has no "
                    f"hard residue rule for active position {int(position)}"
                )
            raw_allowed = raw_rule.get("hard_allowed_residues")
            if (
                isinstance(raw_allowed, (str, bytes))
                or not isinstance(raw_allowed, Sequence)
                or not raw_allowed
            ):
                raise ExecutableNodeCompileError(
                    f"case-owned hard_allowed_residues must be a non-empty list "
                    f"for {segment_name}:{int(position)}"
                )
            allowed_set = {
                str(residue).strip().upper() for residue in raw_allowed
            }
            invalid = sorted(
                residue
                for residue in allowed_set
                if len(residue) != 1 or residue not in canonical
            )
            if invalid:
                raise ExecutableNodeCompileError(
                    f"case-owned hard residue rule contains non-canonical values "
                    f"for {segment_name}:{int(position)}: {invalid}"
                )
            forbidden_set = {
                str(residue).strip().upper()
                for residue in (
                    raw_rule.get("hard_forbidden_residues") or []
                )
            }
            invalid_forbidden = sorted(
                residue
                for residue in forbidden_set
                if len(residue) != 1 or residue not in canonical
            )
            if invalid_forbidden:
                raise ExecutableNodeCompileError(
                    f"case-owned hard forbidden rule contains non-canonical "
                    f"values for {segment_name}:{int(position)}: "
                    f"{invalid_forbidden}"
                )
            allowed = [
                residue
                for residue in canonical
                if residue in allowed_set and residue not in forbidden_set
            ]
            if not allowed:
                raise ExecutableNodeCompileError(
                    f"case-owned hard residue rule leaves no allowed residues "
                    f"for {segment_name}:{int(position)}"
                )

            existing_rule = merged_rules.get(
                str(int(position)),
                merged_rules.get(int(position), {}),
            )
            if existing_rule is not None and not isinstance(existing_rule, Mapping):
                raise ExecutableNodeCompileError(
                    f"node policy position rule must be a mapping for "
                    f"{segment_name}:{int(position)}"
                )
            merged_rule = deepcopy(dict(existing_rule or {}))
            merged_rule["hard_allowed_residues"] = list(allowed)
            if forbidden_set:
                merged_rule["hard_forbidden_residues"] = [
                    residue for residue in canonical if residue in forbidden_set
                ]
            else:
                merged_rule.pop("hard_forbidden_residues", None)
            merged_rules.pop(int(position), None)
            merged_rules[str(int(position))] = merged_rule
            chain_contract = mutation_contract.setdefault(
                str(contract.chain_id), {}
            )
            if int(position) in chain_contract:
                raise ExecutableNodeCompileError(
                    f"case-owned residue contract repeats "
                    f"{contract.chain_id}:{int(position)}"
                )
            chain_contract[int(position)] = list(allowed)
            segment_resolution[str(int(position))] = list(allowed)

        merged_policy["position_residue_rules"] = merged_rules
        merged_policy["case_owned_residue_policy_tier"] = requested_tier
        effective_policies[segment_name] = merged_policy
        resolution["active_segments"][segment_name] = {
            "structural_node_id": str(contract.node_id),
            "chain_id": str(contract.chain_id),
            "positions": segment_resolution,
        }

    if not mutation_contract:
        raise ExecutableNodeCompileError(
            "case-owned residue policy resolved no active editable positions"
        )
    updated["node_edit_policies"] = effective_policies
    updated["residue_mutation_contract"] = mutation_contract
    resolution["residue_mutation_contract"] = deepcopy(mutation_contract)
    updated["case_owned_residue_policy_resolution"] = resolution
    return updated


__all__ = [
    "EXECUTABLE_NODE_SET_VERSION",
    "ExecutableNodeCompileError",
    "ExecutableNodePlan",
    "MeasurementIntent",
    "StructuralNodeContract",
    "compile_executable_ast_nodes",
    "merge_node_policy_patches",
    "apply_case_owned_residue_policy",
]
