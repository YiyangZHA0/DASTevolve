

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple


REGISTRY_VERSION = "astevolve.operator_registry.v2"
EXECUTABLE_STATUSES = frozenset({"enabled", "experimental"})

NODE_HANDLER_IDS = frozenset(
    {
        "point_substitution",
        "block_substitution",
        "segment_resample",
        "site_resample",
        "segment_mutagenesis",
        "motif_graft",
        "region_shuffle",
        "swap",
        "cdr_resample",
    }
)
LEGACY_HANDLER_IDS = frozenset(
    {
        "point_substitution",
        "block_substitution",
        "segment_resample",
        "site_resample",
        "segment_mutagenesis",
        "region_shuffle",
        "swap",
    }
)


class OperatorConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorSpec:


    name: str
    status: str
    node_handler: Optional[str]
    legacy_handler: Optional[str]
    node_kinds: Tuple[str, ...]
    write_set: str
    default_weight: float
    base_weight: float = 0.0
    aliases: Tuple[str, ...] = ()
    description: str = ""


def _spec(
    name: str,
    *,
    node_handler: Optional[str],
    legacy_handler: Optional[str],
    write_set: str,
    default_weight: float,
    base_weight: float = 0.0,
    status: str = "enabled",
    node_kinds: Tuple[str, ...] = ("*",),
    aliases: Tuple[str, ...] = (),
    description: str = "",
) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        status=status,
        node_handler=node_handler,
        legacy_handler=legacy_handler,
        node_kinds=node_kinds,
        write_set=write_set,
        default_weight=float(default_weight),
        base_weight=float(base_weight),
        aliases=aliases,
        description=description,
    )


OPERATOR_REGISTRY: Dict[str, OperatorSpec] = {
    "point": _spec(
        "point",
        node_handler="point_substitution",
        legacy_handler="point_substitution",
        write_set="residue_substitutions",
        default_weight=0.42,
        base_weight=0.55,
    ),
    "block": _spec(
        "block",
        node_handler="block_substitution",
        legacy_handler="block_substitution",
        write_set="contiguous_residue_substitutions",
        default_weight=0.12,
        base_weight=0.14,
    ),
    "segment_resample": _spec(
        "segment_resample",
        node_handler="segment_resample",
        legacy_handler="segment_resample",
        write_set="segment_residue_substitutions",
        default_weight=0.08,
        base_weight=0.08,
    ),
    "site_resample": _spec(
        "site_resample",
        node_handler="site_resample",
        legacy_handler="site_resample",
        write_set="site_biased_residue_substitutions",
        default_weight=0.07,
        base_weight=0.08,
    ),
    "segment_mutagenesis": _spec(
        "segment_mutagenesis",
        node_handler="segment_mutagenesis",
        legacy_handler="segment_mutagenesis",
        write_set="multi_residue_substitutions",
        default_weight=0.08,
        base_weight=0.07,
    ),
    "motif_graft": _spec(
        "motif_graft",
        node_handler="motif_graft",
        legacy_handler=None,
        write_set="contiguous_motif_substitutions",
        default_weight=0.0,
        base_weight=0.0,
        description="Node-aware only and enabled explicitly when a node policy provides a motif source.",
    ),
    "region_shuffle": _spec(
        "region_shuffle",
        node_handler="region_shuffle",
        legacy_handler="region_shuffle",
        write_set="residue_permutation",
        default_weight=0.04,
        base_weight=0.03,
    ),
    "swap": _spec(
        "swap",
        node_handler="swap",
        legacy_handler="swap",
        write_set="two_residue_permutation",
        default_weight=0.04,
        base_weight=0.03,
    ),
    "cdr_resample": _spec(
        "cdr_resample",
        node_handler="cdr_resample",
        legacy_handler=None,
        write_set="cdr_residue_substitutions",
        default_weight=0.05,
        node_kinds=("cdr",),
    ),
    "pocket_motif_swap": _spec(
        "pocket_motif_swap",
        node_handler=None,
        legacy_handler=None,
        write_set="deprecated_pocket_motif_graft_name",
        default_weight=0.0,
        status="unsupported",
        node_kinds=("pocket", "ligand", "loop", "turn"),
        description="Deprecated: this handler performed a motif graft, not a swap; use motif_graft with an explicit motif source.",
    ),
    "negative_design_site_resample": _spec(
        "negative_design_site_resample",
        node_handler=None,
        legacy_handler=None,
        write_set="negative_design_state_consumption_unimplemented",
        default_weight=0.0,
        status="unsupported",
        description="Deprecated: the implementation did not consume signed target/off-target state; use site_resample while negative-design semantics remain in the objective.",
    ),
    "peptide_position_module_mutation": _spec(
        "peptide_position_module_mutation",
        node_handler=None,
        legacy_handler=None,
        write_set="peptide_module_semantics_unimplemented",
        default_weight=0.0,
        status="unsupported",
        description="Deprecated: the implementation was a generic local resample; use site_resample or segment_mutagenesis.",
    ),
    "coupled_betaB_alphaB_mutation": _spec(
        "coupled_betaB_alphaB_mutation",
        node_handler=None,
        legacy_handler=None,
        write_set="cross_node_coupled_substitutions_unimplemented",
        default_weight=0.0,
        status="unsupported",
        description="Deprecated: the implementation modified one node only; use segment_mutagenesis until a true cross-node atomic operator exists.",
    ),
    "linker_length_perturb": _spec(
        "linker_length_perturb",
        node_handler=None,
        legacy_handler=None,
        write_set="sequence_resize_unimplemented",
        default_weight=0.0,
        status="unsupported",
        node_kinds=("linker",),
        description="Deprecated: the implementation did not change sequence length; use the AST length compiler or segment_resample.",
    ),
    "domain_length_perturb": _spec(
        "domain_length_perturb",
        node_handler=None,
        legacy_handler=None,
        write_set="sequence_resize_unimplemented",
        default_weight=0.0,
        status="unsupported",
        description="Deprecated: the implementation did not change sequence length; use the AST length compiler or segment_mutagenesis.",
    ),
    "coupled_cdr_resample": _spec(
        "coupled_cdr_resample",
        node_handler=None,
        legacy_handler=None,
        write_set="cross_node_cdr_substitutions_unimplemented",
        default_weight=0.0,
        status="unsupported",
        node_kinds=("cdr",),
        description="Evaluator recommendation has no cross-CDR execution handler.",
    ),
}


def _alias_index() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for name, spec in OPERATOR_REGISTRY.items():
        if spec.status not in {"enabled", "experimental", "unsupported"}:
            raise RuntimeError(f"Invalid operator status for {name}: {spec.status}")
        if spec.node_handler is not None and spec.node_handler not in NODE_HANDLER_IDS:
            raise RuntimeError(f"Unknown node handler for {name}: {spec.node_handler}")
        if spec.legacy_handler is not None and spec.legacy_handler not in LEGACY_HANDLER_IDS:
            raise RuntimeError(f"Unknown legacy handler for {name}: {spec.legacy_handler}")
        for alias in spec.aliases:
            if alias in OPERATOR_REGISTRY or alias in aliases:
                raise RuntimeError(f"Duplicate operator alias: {alias}")
            aliases[alias] = name
    return aliases


OPERATOR_ALIASES = _alias_index()


def get_operator_spec(name: Any) -> OperatorSpec:


    raw = str(name)
    canonical = OPERATOR_ALIASES.get(raw, raw)
    try:
        return OPERATOR_REGISTRY[canonical]
    except KeyError as exc:
        raise OperatorConfigError(f"Unknown mutation operator {raw!r}") from exc


def require_operator(name: Any, mode: str = "node") -> OperatorSpec:


    spec = get_operator_spec(name)
    if spec.status == "unsupported":
        raise OperatorConfigError(
            f"Mutation operator {spec.name!r} is unsupported: {spec.description}"
        )
    if mode == "node":
        handler = spec.node_handler
    elif mode == "legacy":
        handler = spec.legacy_handler
    else:
        raise ValueError(f"Unknown operator execution mode {mode!r}")
    if handler is None:
        raise OperatorConfigError(
            f"Mutation operator {spec.name!r} is not supported by the {mode} executor"
        )
    return spec


def operator_supports_node_kind(name: Any, node_kind: Any) -> bool:


    spec = get_operator_spec(name)
    allowed = {str(kind).lower() for kind in spec.node_kinds}
    return "*" in allowed or str(node_kind or "").lower() in allowed


def executable_operator_names(mode: str = "node") -> frozenset[str]:


    names = []
    for name, spec in OPERATOR_REGISTRY.items():
        if spec.status not in EXECUTABLE_STATUSES:
            continue
        handler = spec.node_handler if mode == "node" else spec.legacy_handler if mode == "legacy" else None
        if mode not in {"node", "legacy"}:
            raise ValueError(f"Unknown operator execution mode {mode!r}")
        if handler is not None:
            names.append(name)
    return frozenset(names)


def validate_operator_weights(
    weights: Mapping[str, Any],
    *,
    mode: str = "node",
    context: str = "mutation_ops",
    drop_zero: bool = False,
) -> Dict[str, float]:


    if not isinstance(weights, Mapping):
        raise OperatorConfigError(f"{context} must be a mapping of operator names to weights")
    out: Dict[str, float] = {}
    for raw_name, raw_weight in weights.items():
        try:
            spec = require_operator(raw_name, mode=mode)
        except OperatorConfigError as exc:
            raise OperatorConfigError(f"{context}: {exc}") from exc
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise OperatorConfigError(
                f"{context}.{raw_name} has non-numeric weight {raw_weight!r}"
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise OperatorConfigError(
                f"{context}.{raw_name} weight must be finite and non-negative"
            )
        if not drop_zero or weight > 0.0:
            out[spec.name] = out.get(spec.name, 0.0) + weight
    if not out or sum(out.values()) <= 0.0:
        raise OperatorConfigError(f"{context} must contain at least one positive operator weight")
    return {
        name: out[name]
        for name in OPERATOR_REGISTRY
        if name in out
    }


def default_operator_weights(
    mode: str = "node",
    *,
    profile: str = "runtime",
) -> Dict[str, float]:


    names = executable_operator_names(mode)
    if profile not in {"base", "runtime"}:
        raise ValueError(f"Unknown operator-weight profile {profile!r}")
    return {
        name: (
            OPERATOR_REGISTRY[name].base_weight
            if profile == "base"
            else OPERATOR_REGISTRY[name].default_weight
        )
        for name in OPERATOR_REGISTRY
        if name in names
        and (
            OPERATOR_REGISTRY[name].base_weight
            if profile == "base"
            else OPERATOR_REGISTRY[name].default_weight
        )
        > 0.0
    }


_PROPOSAL_TIER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "exploit": {
        "point": 0.40,
        "site_resample": 0.34,
        "block": 0.18,
        "swap": 0.08,
    },
    "explore": {
        "segment_mutagenesis": 0.38,
        "motif_graft": 0.24,
        "block": 0.20,
        "site_resample": 0.10,
        "region_shuffle": 0.08,
    },
    "repair": {"point": 0.58, "site_resample": 0.30, "swap": 0.12},
    "frozen": {},
}


def proposal_tier_operator_weights(tier: str) -> Dict[str, float]:


    name = str(tier).lower()
    if name == "legacy":
        return default_operator_weights("node")
    if name not in _PROPOSAL_TIER_WEIGHTS:
        raise ValueError(f"Unknown proposal tier {tier!r}")
    raw = _PROPOSAL_TIER_WEIGHTS[name]
    if not raw:
        return {}
    return validate_operator_weights(
        raw,
        mode="node",
        context=f"proposal_tier.{name}",
    )


def operator_manifest(name: Any, mode: str = "node") -> Dict[str, Any]:


    spec = require_operator(name, mode=mode)
    handler = spec.node_handler if mode == "node" else spec.legacy_handler
    return {
        "registry_version": REGISTRY_VERSION,
        "name": spec.name,
        "status": spec.status,
        "handler": handler,
        "node_kinds": list(spec.node_kinds),
        "write_set": spec.write_set,
    }


__all__ = [
    "EXECUTABLE_STATUSES",
    "LEGACY_HANDLER_IDS",
    "NODE_HANDLER_IDS",
    "OPERATOR_ALIASES",
    "OPERATOR_REGISTRY",
    "OperatorConfigError",
    "OperatorSpec",
    "REGISTRY_VERSION",
    "default_operator_weights",
    "executable_operator_names",
    "get_operator_spec",
    "operator_manifest",
    "operator_supports_node_kind",
    "proposal_tier_operator_weights",
    "require_operator",
    "validate_operator_weights",
]
