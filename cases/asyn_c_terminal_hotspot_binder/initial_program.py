"""Executable hotspot-biased Dual-AST strategy for the ASYN binder case."""

# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Any, Dict

from engine.default_strategy import base_strategy


def propose_strategy() -> Dict[str, Any]:
    """Expose all 50 binder residues through five resizable search nodes."""

    strategy = base_strategy()
    strategy.update(
        {
            "outer_loop_phase": "explore_ast",
            "mcts_c_puct": 1.7,
            "mcts_max_depth": 8,
            "mcts_progressive_widening_c": 0.25,
            "mcts_progressive_widening_alpha": 0.5,
            "ast_revision_plan": {   'schema_version': 'astevolve.ast_revision_plan.v2',
                                     'structural_nodes': [   {   'node_id': 'llm_binder_nterm_seed',
                                                                 'selector': {'schema_version': 'astevolve.residue_selector.v1', 'chain_id': 'B', 'spans': [[0, 9]]},
                                                                 'action_profile': 'point_resample',
                                                                 'intent': 'N-terminal helix cap; C1A proven, G6 suppression for helix nucleation. Stable at 9 residues with proven substitutions. Policy weight moderate since key edits established.',
                                                                 'evidence_refs': ['asyn:B:0', 'asyn:B:1'],
                                                                 'residue_policy': {   'favored_residues': ['A', 'E', 'K', 'L', 'Q', 'R', 'M'],
                                                                                       'disfavored_residues': ['C', 'G', 'P', 'W', 'N', 'S'],
                                                                                       'policy_weight': 1.2,
                                                                                       'position_residue_rules': {   '1': {   'favored_residues': ['A', 'L', 'E', 'K'],
                                                                                                                              'disfavored_residues': ['C', 'G', 'P', 'W'],
                                                                                                                              'policy_weight': 1.8,
                                                                                                                              'intent': 'Encode proven C1A from n14; eliminate cysteine for developability'},
                                                                                                                     '6': {   'favored_residues': ['A', 'L', 'E', 'K', 'Q'],
                                                                                                                              'disfavored_residues': ['G', 'P', 'C', 'W'],
                                                                                                                              'policy_weight': 1.6,
                                                                                                                              'intent': 'Replace helix-breaking G6 to restore N-terminal helix continuity'}}}},
                                                             {   'node_id': 'llm_binder_upper_seed',
                                                                 'selector': {'schema_version': 'astevolve.residue_selector.v1', 'chain_id': 'B', 'spans': [[9, 19]]},
                                                                 'action_profile': 'point_resample',
                                                                 'intent': 'Developability node at 10 residues. Coordinates W11/W14 aromatic elimination and helix-breaker suppression. n95 S13L/R16E/I18Q success validates multi-site intervention.',
                                                                 'evidence_refs': ['asyn:B:10', 'asyn:B:11'],
                                                                 'residue_policy': {   'favored_residues': ['A', 'E', 'K', 'L', 'Q', 'R', 'M'],
                                                                                       'disfavored_residues': ['C', 'G', 'P', 'W'],
                                                                                       'policy_weight': 1.4,
                                                                                       'position_residue_rules': {   '11': {   'favored_residues': ['L', 'M', 'A', 'Q'],
                                                                                                                               'disfavored_residues': ['W', 'C', 'G', 'P'],
                                                                                                                               'policy_weight': 1.7,
                                                                                                                               'intent': 'Replace W11 aromatic with helix-compatible hydrophobic to reduce aggregation risk'},
                                                                                                                     '13': {   'favored_residues': ['L', 'M', 'A', 'E'],
                                                                                                                               'disfavored_residues': ['S', 'G', 'P', 'W'],
                                                                                                                               'policy_weight': 1.3,
                                                                                                                               'intent': 'Encode n95-proven S13L helix-stabilizing substitution in developability cluster'},
                                                                                                                     '14': {   'favored_residues': ['L', 'M', 'A', 'Q', 'E'],
                                                                                                                               'disfavored_residues': ['W', 'C', 'G', 'P'],
                                                                                                                               'policy_weight': 1.6,
                                                                                                                               'intent': 'Replace W14 aromatic burden; second tryptophan in developability cluster'}}}},
                                                             {   'node_id': 'llm_binder_center_seed',
                                                                 'selector': {'schema_version': 'astevolve.residue_selector.v1', 'chain_id': 'B', 'spans': [[19, 31]]},
                                                                 'action_profile': 'point_resample',
                                                                 'intent': 'Primary interface and helix-junction node at 12 residues spanning B[19,31). Captures P-SEA helix bridge B23-B29 with Pro breakers at keys 23/28. Added rules for keys 27 and 29 complete '
                                                                           'helix-bridge coverage. M19 and K30 flanking anchors provide helix-compatible context. Highest policy_weight 1.9 for prioritized interface helix-continuity search.',
                                                                 'evidence_refs': ['asyn:B:20', 'asyn:B:21'],
                                                                 'residue_policy': {   'favored_residues': ['A', 'E', 'K', 'L', 'M', 'Q', 'R', 'I', 'V'],
                                                                                       'disfavored_residues': ['C', 'G', 'P', 'W'],
                                                                                       'policy_weight': 1.9,
                                                                                       'position_residue_rules': {   '23': {   'favored_residues': ['A', 'L', 'E', 'K', 'Q'],
                                                                                                                               'disfavored_residues': ['P', 'G', 'C', 'W'],
                                                                                                                               'policy_weight': 1.9,
                                                                                                                               'intent': 'Replace P23 helix breaker to restore central helix continuity per P-SEA bridge diagnostic'},
                                                                                                                     '24': {   'favored_residues': ['L', 'A', 'E', 'K', 'M'],
                                                                                                                               'disfavored_residues': ['C', 'G', 'P', 'W'],
                                                                                                                               'policy_weight': 1.6,
                                                                                                                               'intent': 'Eliminate C24 cysteine for developability; maintain interface potential'},
                                                                                                                     '25': {   'favored_residues': ['L', 'I', 'V', 'M', 'A'],
                                                                                                                               'disfavored_residues': ['C', 'G', 'P', 'W'],
                                                                                                                               'policy_weight': 1.5,
                                                                                                                               'intent': 'n86 L25Y validates interface hydrophobic tuning; favor helix-compatible hydrophobics for amphipathic face'},
                                                                                                                     '26': {   'favored_residues': ['A', 'L', 'E', 'K', 'Q'],
                                                                                                                               'disfavored_residues': ['G', 'P', 'C', 'W'],
                                                                                                                               'policy_weight': 1.4,
                                                                                                                               'intent': 'Replace G26 helix breaker in interface region; n51 A26Q validated'},
                                                                                                                     '27': {   'favored_residues': ['A', 'L', 'E', 'K', 'Q', 'M'],
                                                                                                                               'disfavored_residues': ['G', 'P', 'C', 'W'],
                                                                                                                               'policy_weight': 1.7,
                                                                                                                               'intent': 'Helix bridge interior between P23 and P28 breakers; stabilize with helix-compatible residues to support continuous alpha geometry '
                                                                                                                                         'across B23-B29'},
                                                                                                                     '28': {   'favored_residues': ['A', 'L', 'E', 'K', 'Q'],
                                                                                                                               'disfavored_residues': ['P', 'G', 'C', 'W'],
                                                                                                                               'policy_weight': 1.9,
                                                                                                                               'intent': 'Replace P28 helix breaker to restore helix through interface node per P-SEA bridge diagnostic'},
                                                                                                                     '29': {   'favored_residues': ['A', 'L', 'E', 'K', 'M', 'Q'],
                                                                                                                               'disfavored_residues': ['G', 'P', 'C', 'W'],
                                                                                                                               'policy_weight': 1.7,
                                                                                                                               'intent': 'Nucleation site for second alpha segment B29-B49 per P-SEA; helix-compatible residues stabilize the C-terminal helix start for '
                                                                                                                                         'interface engagement'}}}},
                                                             {   'node_id': 'llm_binder_lower_seed',
                                                                 'selector': {'schema_version': 'astevolve.residue_selector.v1', 'chain_id': 'B', 'spans': [[31, 40]]},
                                                                 'action_profile': 'point_resample',
                                                                 'intent': 'Developability core at 9 residues. G38 suppression, proven V33K/T36L encoded. n68 D31M/I34N/N39M validates this region. n88 L37Q fast-loss improvement supports continued exploration.',
                                                                 'evidence_refs': ['asyn:B:30', 'asyn:B:31'],
                                                                 'residue_policy': {   'favored_residues': ['A', 'E', 'K', 'L', 'Q', 'R', 'M'],
                                                                                       'disfavored_residues': ['C', 'G', 'P', 'W'],
                                                                                       'policy_weight': 1.2,
                                                                                       'position_residue_rules': {   '33': {   'favored_residues': ['K', 'F', 'L'],
                                                                                                                               'disfavored_residues': ['V', 'G', 'P'],
                                                                                                                               'policy_weight': 1.3,
                                                                                                                               'intent': 'Encode proven V33K/F from candidates n33/n48'},
                                                                                                                     '36': {   'favored_residues': ['L', 'M', 'A'],
                                                                                                                               'disfavored_residues': ['T', 'G', 'P'],
                                                                                                                               'policy_weight': 1.2,
                                                                                                                               'intent': 'Encode proven T36L helix-stabilizing substitution from n33'},
                                                                                                                     '38': {   'favored_residues': ['A', 'L', 'E', 'K', 'Q'],
                                                                                                                               'disfavored_residues': ['G', 'P', 'C', 'W'],
                                                                                                                               'policy_weight': 1.5,
                                                                                                                               'intent': 'Replace G38 helix breaker to restore C-terminal helix continuity'}}}},
                                                             {   'node_id': 'llm_binder_cterm_seed',
                                                                 'selector': {'schema_version': 'astevolve.residue_selector.v1', 'chain_id': 'B', 'spans': [[40, 50]]},
                                                                 'action_profile': 'point_resample',
                                                                 'intent': 'C-terminal hotspot localization at 10 residues. Captures proven I40R alongside I48K for dual charge-anchor hotspot engagement. Added key 42 rule to stabilize B29-B49 alpha segment interior for '
                                                                           'improved interface pLDDT. pLDDT improved from 28.5 to 44.1.',
                                                                 'evidence_refs': ['asyn:B:40', 'asyn:B:41'],
                                                                 'residue_policy': {   'favored_residues': ['A', 'E', 'K', 'L', 'Q', 'R', 'M'],
                                                                                       'disfavored_residues': ['C', 'G', 'P', 'W'],
                                                                                       'policy_weight': 1.4,
                                                                                       'position_residue_rules': {   '40': {   'favored_residues': ['R', 'K', 'E'],
                                                                                                                               'disfavored_residues': ['I', 'V', 'G', 'P'],
                                                                                                                               'policy_weight': 1.4,
                                                                                                                               'intent': 'Encode proven I40R from n32 for charged hotspot contact with acidic ASYN C-terminus'},
                                                                                                                     '42': {   'favored_residues': ['L', 'M', 'A', 'K', 'Q'],
                                                                                                                               'disfavored_residues': ['G', 'P', 'C', 'W'],
                                                                                                                               'policy_weight': 1.4,
                                                                                                                               'intent': 'Stabilize B29-B49 alpha segment interior; helix-compatible residues improve local pLDDT and interface contact confidence near ASYN '
                                                                                                                                         'C-terminal hotspot'},
                                                                                                                     '48': {   'favored_residues': ['K', 'R', 'Q'],
                                                                                                                               'disfavored_residues': ['I', 'V', 'G', 'P'],
                                                                                                                               'policy_weight': 1.3,
                                                                                                                               'intent': 'Encode proven I48K from n47 for terminal charge complementarity with ASYN'}}}}],
                                     'mapping_edges': [   {'edge_id': 'llm_nterm_holo_point', 'functional_node_id': 'binder_holo_fold', 'structural_node_id': 'llm_binder_nterm_seed', 'action_operator': 'point', 'evidence_refs': ['asyn:B:0', 'asyn:B:1']},
                                                          {   'edge_id': 'llm_upper_developability_resample',
                                                              'functional_node_id': 'developability',
                                                              'structural_node_id': 'llm_binder_upper_seed',
                                                              'action_operator': 'site_resample',
                                                              'evidence_refs': ['asyn:B:10', 'asyn:B:11']},
                                                          {'edge_id': 'llm_center_interface_point', 'functional_node_id': 'hotspot_interface', 'structural_node_id': 'llm_binder_center_seed', 'action_operator': 'point', 'evidence_refs': ['asyn:B:20', 'asyn:B:21']},
                                                          {'edge_id': 'llm_lower_developability_point', 'functional_node_id': 'developability', 'structural_node_id': 'llm_binder_lower_seed', 'action_operator': 'point', 'evidence_refs': ['asyn:B:30', 'asyn:B:31']},
                                                          {   'edge_id': 'llm_cterm_localization_resample',
                                                              'functional_node_id': 'hotspot_localization',
                                                              'structural_node_id': 'llm_binder_cterm_seed',
                                                              'action_operator': 'site_resample',
                                                              'evidence_refs': ['asyn:B:40', 'asyn:B:41']}],
                                     'decision_record': {   'action': 'revise',
                                                            'diagnosis': 'Interface quality residual 0.475 and holo_confidence residual 0.483 are the dominant energy contributors. P-SEA helix bridge B23-B29 has Pro breakers at keys 23/28 but positions 27 and 29 within '
                                                                         'the bridge lack explicit stabilization rules. The second alpha segment nucleation at key 29 is unaddressed. C-terminal node key 42 in the B29-B49 alpha segment lacks a helix-stability rule despite '
                                                                         'interface pLDDT being only 0.33.',
                                                            'hypothesis': 'Completing position-level helix-continuity rules across the P-SEA bridge (keys 27, 29) and stabilizing the second alpha segment (key 42) will raise interface_plddt_mean from 0.33 toward 0.38+ and '
                                                                          'reduce asyn_interface_quality residual from 0.475 by improving local fold confidence at the hotspot contact interface. The center node at policy_weight 1.9 prioritizes this search.',
                                                            'evidence_refs': ['asyn:B:0', 'asyn:B:1', 'asyn:B:10', 'asyn:B:11', 'asyn:B:20', 'asyn:B:21', 'asyn:B:30', 'asyn:B:31', 'asyn:B:40', 'asyn:B:41'],
                                                            'expected_effects': [   'interface_plddt_mean rises from 0.33 toward 0.38 as helix bridge positions 27/29 gain explicit stabilization',
                                                                                    'asyn_interface_quality residual decreases from 0.475 as helix continuity improves contact geometry',
                                                                                    'asyn_binder_holo_confidence residual decreases from 0.483 with improved B03 local pLDDT',
                                                                                    'hotspot_localization maintained above 0.94 with unchanged C-terminal charge-anchor rules',
                                                                                    'Developability maintained above 0.86 with unchanged cysteine/aromatic suppression'],
                                                            'failure_condition': 'Reject if final_energy exceeds 0.30, asyn_interface_quality drops below 0.45, B03 pLDDT_mean drops below 42, developability drops below 0.80, or hard_gate_pass becomes false.',
                                                            'confidence': 0.78,
                                                            'rationale': 'Old lengths [9,10,12,9,10] sum=50. New lengths [9,10,12,9,10] sum=50 (unchanged partition). Policy revision targets interface_quality residual 0.475 and holo_confidence residual 0.483. P-SEA '
                                                                         'identifies helix bridge B23-B29 with Pro breakers at keys 23/28; positions 27 and 29 within this bridge lack explicit rules. Adding rules for keys 27 (helix bridge interior) and 29 (second alpha '
                                                                         'segment nucleation per P-SEA B29-B49) completes helix-continuity coverage. C-terminal key 42 rule stabilizes the B29-B49 alpha segment for hotspot engagement. Center policy_weight raised 1.8→1.9 as '
                                                                         'highest-priority interface node. B01 pLDDT 38.5 remains lowest but stable; B04 pLDDT 49.3 is highest. n6 P23A tested without improvement, confirming need for multi-position helix stabilization '
                                                                         'rather than single-site edits.',
                                                            'rollback_condition': 'Revert to parent policy weights and remove keys 27/29/42 position rules if interface_quality worsens below 0.50 or center node pLDDT drops below 44 after one iteration.'}},
            "layout_plan": {
                "binder_domain_order": ["binder_domain"],
                "design_regions": [
                    {
                        "name": f"binder_tile_{tile_index:02d}",
                        "bind_to": [f"B{tile_index:02d}_tile_{tile_index:02d}"],
                        "favored_residues": ["A", "E", "K", "L", "M", "Q", "R"],
                        "disfavored_residues": ["C", "G", "P", "W"],
                        "operator_phase": "explore",
                        "mutation_rate": 0.10,
                        "mutation_ops": {"point": 0.65, "site_resample": 0.35},
                    }
                    for tile_index in range(1, 6)
                ],
            },
        }
    )
    return strategy


# EVOLVE-BLOCK-END

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Optional

from astevolve.runtime.case_context import current_case_kwargs
from astevolve.runtime.case_program import merge_locked_runtime_defaults
from astevolve.runtime.paths import artifact_path
from engine.case_builder import prepare_case_inputs, run_design_search
from engine.runtime_profile import build_sa_config

CASE_ID = "asyn_c_terminal_hotspot_binder"


def _case_root() -> Path:
    manifest = os.environ.get("ASTEVOLVE_CASE_MANIFEST")
    return Path(manifest).resolve().parent if manifest else Path(__file__).resolve().parent


if str(_case_root()) not in sys.path:
    sys.path.insert(0, str(_case_root()))
from asyn_binder_evaluator import register_asyn_hotspot_binder_plugin
register_asyn_hotspot_binder_plugin()


def _protocol() -> Dict[str, Any]:
    return json.loads((_case_root() / "protocol.json").read_text(encoding="utf-8"))


def _locked_runtime() -> Dict[str, Any]:
    locked = json.loads((_case_root() / "runtime_config.json").read_text(encoding="utf-8"))
    model = os.environ.get("ASTEVOLVE_ESMFOLD_MODEL", "facebook/esmfold_v1")
    locked["inner_structure_model_name"] = model
    locked["structure_model_name"] = model
    return locked



def _dynamic_partition_contract() -> Dict[str, Any] | None:
    value = _protocol().get("dynamic_node_partition")
    return deepcopy(dict(value)) if isinstance(value, Mapping) else None


def _partition_rows(strategy: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Return position-ordered logical nodes after enforcing the case contract."""

    contract = _dynamic_partition_contract()
    if contract is None:
        return []
    if contract.get("schema_version") != (
        "astevolve.structured_ast_partition_contract.v1"
    ):
        raise RuntimeError("unsupported ASYN dynamic-node partition contract")
    plan = strategy.get("ast_revision_plan")
    nodes = plan.get("structural_nodes") if isinstance(plan, Mapping) else None
    node_count = int(contract["node_count"])
    if not isinstance(nodes, list) or len(nodes) != node_count:
        raise RuntimeError(
            f"ASYN dynamic partition requires exactly {node_count} structural nodes"
        )
    rows: list[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise RuntimeError("ASYN dynamic partition node must be a mapping")
        selector = node.get("selector")
        spans = selector.get("spans") if isinstance(selector, Mapping) else None
        if (
            not isinstance(selector, Mapping)
            or selector.get("chain_id") != contract["chain_id"]
            or not isinstance(spans, list)
            or len(spans) != 1
            or not isinstance(spans[0], (list, tuple))
            or len(spans[0]) != 2
        ):
            raise RuntimeError(
                "every ASYN dynamic partition node must have one contiguous B span"
            )
        start, end = spans[0]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise RuntimeError("ASYN dynamic partition boundaries must be integers")
        length = int(end) - int(start)
        if not (
            int(contract["min_positions_per_node"])
            <= length
            <= int(contract["max_positions_per_node"])
        ):
            raise RuntimeError("ASYN dynamic partition node length is out of bounds")
        rows.append(
            {
                "node_id": str(node.get("node_id") or ""),
                "start": int(start),
                "end": int(end),
                "length": length,
                "residue_policy": deepcopy(dict(node.get("residue_policy") or {})),
            }
        )
    rows.sort(key=lambda row: (row["start"], row["end"], row["node_id"]))
    cursor = int(contract["start"])
    for row in rows:
        if not row["node_id"] or row["start"] != cursor:
            raise RuntimeError(
                "ASYN dynamic partition must be adjacent with no gap or overlap"
            )
        cursor = row["end"]
    if cursor != int(contract["end"]):
        raise RuntimeError("ASYN dynamic partition does not cover all 50 residues")
    return rows


def _apply_dynamic_partition_layout(
    strategy: Mapping[str, Any], rows: list[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Map LLM node emphasis onto the five stable physical search segments."""

    updated = deepcopy(dict(strategy))
    if not rows:
        return updated
    physical_names = [f"B{index:02d}_tile_{index:02d}" for index in range(1, 6)]
    regions = []
    for physical_name, row in zip(physical_names, rows):
        residue_policy = row.get("residue_policy")
        residue_policy = residue_policy if isinstance(residue_policy, Mapping) else {}
        raw_weight = residue_policy.get("policy_weight", 1.0)
        weight = (
            float(raw_weight)
            if isinstance(raw_weight, (int, float))
            and not isinstance(raw_weight, bool)
            else 1.0
        )
        weight = max(0.05, min(5.0, weight))
        regions.append(
            {
                "name": f"dynamic_{physical_name}",
                "bind_to": [physical_name],
                "favored_residues": list(
                    residue_policy.get("favored_residues") or []
                ),
                "disfavored_residues": list(
                    residue_policy.get("disfavored_residues") or []
                ),
                "operator_phase": "explore",
                "mutation_rate": 0.10,
                "mutation_ops": {"point": 0.65, "site_resample": 0.35},
                "node_weights": {physical_name: weight},
                "role": (
                    f"Logical node {row['node_id']} owns B[{row['start']},"
                    f"{row['end']}) in this evaluation."
                ),
            }
        )
    layout = deepcopy(dict(updated.get("layout_plan") or {}))
    layout["binder_domain_order"] = ["binder_domain"]
    layout["design_regions"] = regions
    updated["layout_plan"] = layout
    return updated


def _materialize_dynamic_design_state(
    strategy: Mapping[str, Any], paths: Mapping[str, str]
) -> Dict[str, str]:
    """Resegment, but never mutate, the fixed 50-aa Bagel starting sequence."""

    rows = _partition_rows(strategy)
    if not rows:
        return dict(paths)
    source_path = Path(paths["design_state_path"]).resolve()
    state = json.loads(source_path.read_text(encoding="utf-8"))
    state["case_sheet_path"] = str(source_path.parent / "case_sheet.json")
    state["memory_path"] = str(source_path.parent / "memory.yaml")
    binder = state["binder"]
    parts_key = next(iter(binder["domain_segment_keys"].values()))
    source_parts = binder[parts_key]
    if (
        not isinstance(source_parts, list)
        or len(source_parts) != len(rows)
        or len(rows) != 5
    ):
        raise RuntimeError("dynamic partition requires exactly five source segments")
    sequence = "".join(str(part[2]) for part in source_parts)
    expected_sequence = _protocol()["fixed_inputs"]["initial_binder_sequence"]
    if sequence != expected_sequence or len(sequence) != int(rows[-1]["end"]):
        raise RuntimeError(
            "dynamic partition source must equal the fixed Bagel state-0 sequence"
        )

    physical_names = [f"B{index:02d}_tile_{index:02d}" for index in range(1, 6)]
    kinds = [str(part[1]) for part in source_parts]
    binder[parts_key] = [
        [name, kind, sequence[int(row["start"]) : int(row["end"])]]
        for name, kind, row in zip(physical_names, kinds, rows)
    ]
    binder["architecture"] = "bagel_state0_dynamic_five_node_partition"

    constraints = state["design_constraints"]
    constraints["mutable_residue_spans"] = [
        {
            "chain_id": "B",
            "node": name,
            "spans": [[int(row["start"]), int(row["end"])]],
        }
        for name, row in zip(physical_names, rows)
    ]
    policy = state["global_ast_evolution_policy"]
    contract = _dynamic_partition_contract() or {}
    total_length = int(contract["end"]) - int(contract["start"])
    policy.update(
        {
            "min_total_editable_positions": total_length,
            "max_total_editable_positions": total_length,
            "min_active_structural_nodes": int(contract["node_count"]),
            "max_active_structural_nodes": int(contract["node_count"]),
            "max_positions_per_node": int(contract["max_positions_per_node"]),
            "max_spans_per_node": 1,
        }
    )
    graph_nodes = (
        state.get("semantic_graph", {})
        .get("structural_graph", {})
        .get("nodes", {})
    )
    if isinstance(graph_nodes, dict):
        for name, row in zip(physical_names, rows):
            if isinstance(graph_nodes.get(name), dict):
                graph_nodes[name]["spans"] = [
                    [int(row["start"]), int(row["end"])]
                ]

    state["dynamic_node_partition"] = {
        "schema_version": "astevolve.asyn_dynamic_node_partition.v1",
        "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "logical_nodes": [dict(row) for row in rows],
        "physical_segments": [
            {
                "name": name,
                "span": [int(row["start"]), int(row["end"])],
                "length": int(row["length"]),
                "logical_node_id": str(row["node_id"]),
            }
            for name, row in zip(physical_names, rows)
        ],
        "total_length": len(sequence),
    }
    canonical = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    output_dir = artifact_path(CASE_ID, "compiled_states")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dynamic_design_state_{digest}.json"
    if not output_path.is_file():
        temporary = output_path.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_path)
    updated_paths = dict(paths)
    updated_paths["design_state_path"] = str(output_path)
    return updated_paths

def runtime_strategy() -> Dict[str, Any]:
    proposed = propose_strategy()
    proposed["graph_ablation_mode"] = "full"
    rows = _partition_rows(proposed)
    proposed = _apply_dynamic_partition_layout(proposed, rows)
    return merge_locked_runtime_defaults(proposed, _locked_runtime(), base_strategy)


def _case_paths(strategy: Mapping[str, Any]) -> Dict[str, str]:
    paths = current_case_kwargs(CASE_ID, case_root=_case_root())
    return _materialize_dynamic_design_state(strategy, paths)


def preview_case() -> Dict[str, Any]:
    strategy = runtime_strategy()
    cfg = build_sa_config(strategy)
    prepared = prepare_case_inputs(strategy, mapping_execution_mode="full", **_case_paths(strategy))
    return {
        "case_id": CASE_ID,
        "mapping_execution_mode": "full",
        "initial_sequences": prepared.template_sequences,
        "mask_true_counts": {chain: int(sum(mask)) for chain, mask in prepared.masks.items()},
        "fixed_residue_counts": {chain: len(values) for chain, values in prepared.fixed_residues.items()},
        "inner_budget": cfg["iterations"],
        "outer_evaluations": _protocol()["budget"]["outer_evaluations"],
        "provider": cfg["inner_structure_model"],
        "dynamic_node_partition": deepcopy(_partition_rows(strategy)),
        "executable_structural_node_ids": [node.node_id for node in prepared.executable_node_plan.structural_nodes],
        "measurement_intent_ids": [intent.functional_node_id for intent in prepared.executable_node_plan.measurement_intents],
    }


def run_search(seed: Optional[int] = None) -> Dict[str, Any]:
    strategy = runtime_strategy()
    strategy["mcts_output_dir"] = str(artifact_path(CASE_ID, "inner", f"seed_{int(seed or 0)}"))
    return run_design_search(
        strategy, seed=seed, memory_commit_mode="deferred",
        mapping_execution_mode="full", **_case_paths(strategy),
    )


__all__ = ["preview_case", "propose_strategy", "run_search", "runtime_strategy"]
