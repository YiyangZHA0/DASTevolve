"""Executable hotspot-biased Dual-AST strategy for the ASYN binder case."""

# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Any, Dict

from engine.default_strategy import base_strategy


def propose_strategy() -> Dict[str, Any]:
    """Initialize five equal tiles with no candidate-derived residue policy."""

    strategy = base_strategy()
    strategy.update(
        {'outer_loop_phase': 'explore_ast',
         'mcts_c_puct': 1.7,
         'mcts_max_depth': 8,
         'mcts_progressive_widening_c': 0.25,
         'mcts_progressive_widening_alpha': 0.5,
         'ast_revision_plan': {'schema_version': 'astevolve.ast_revision_plan.v2',
                               'structural_nodes': [{'node_id': 'llm_binder_tile_01',
                                                     'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                                  'chain_id': 'B',
                                                                  'spans': [[0, 10]]},
                                                     'action_profile': 'point_resample',
                                                     'intent': 'Explore this equal-length binder tile '
                                                               'under the fixed task objectives.',
                                                     'evidence_refs': ['asyn:B:0',
                                                                       'asyn:B:1',
                                                                       'asyn:B:2',
                                                                       'asyn:B:3',
                                                                       'asyn:B:4',
                                                                       'asyn:B:5',
                                                                       'asyn:B:6',
                                                                       'asyn:B:7',
                                                                       'asyn:B:8',
                                                                       'asyn:B:9'],
                                                     'residue_policy': {'favored_residues': [],
                                                                        'disfavored_residues': [],
                                                                        'policy_weight': 1.0,
                                                                        'position_residue_rules': {}}},
                                                    {'node_id': 'llm_binder_tile_02',
                                                     'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                                  'chain_id': 'B',
                                                                  'spans': [[10, 20]]},
                                                     'action_profile': 'point_resample',
                                                     'intent': 'Explore this equal-length binder tile '
                                                               'under the fixed task objectives.',
                                                     'evidence_refs': ['asyn:B:10',
                                                                       'asyn:B:11',
                                                                       'asyn:B:12',
                                                                       'asyn:B:13',
                                                                       'asyn:B:14',
                                                                       'asyn:B:15',
                                                                       'asyn:B:16',
                                                                       'asyn:B:17',
                                                                       'asyn:B:18',
                                                                       'asyn:B:19'],
                                                     'residue_policy': {'favored_residues': [],
                                                                        'disfavored_residues': [],
                                                                        'policy_weight': 1.0,
                                                                        'position_residue_rules': {}}},
                                                    {'node_id': 'llm_binder_tile_03',
                                                     'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                                  'chain_id': 'B',
                                                                  'spans': [[20, 30]]},
                                                     'action_profile': 'point_resample',
                                                     'intent': 'Explore this equal-length binder tile '
                                                               'under the fixed task objectives.',
                                                     'evidence_refs': ['asyn:B:20',
                                                                       'asyn:B:21',
                                                                       'asyn:B:22',
                                                                       'asyn:B:23',
                                                                       'asyn:B:24',
                                                                       'asyn:B:25',
                                                                       'asyn:B:26',
                                                                       'asyn:B:27',
                                                                       'asyn:B:28',
                                                                       'asyn:B:29'],
                                                     'residue_policy': {'favored_residues': [],
                                                                        'disfavored_residues': [],
                                                                        'policy_weight': 1.0,
                                                                        'position_residue_rules': {}}},
                                                    {'node_id': 'llm_binder_tile_04',
                                                     'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                                  'chain_id': 'B',
                                                                  'spans': [[30, 40]]},
                                                     'action_profile': 'point_resample',
                                                     'intent': 'Explore this equal-length binder tile '
                                                               'under the fixed task objectives.',
                                                     'evidence_refs': ['asyn:B:30',
                                                                       'asyn:B:31',
                                                                       'asyn:B:32',
                                                                       'asyn:B:33',
                                                                       'asyn:B:34',
                                                                       'asyn:B:35',
                                                                       'asyn:B:36',
                                                                       'asyn:B:37',
                                                                       'asyn:B:38',
                                                                       'asyn:B:39'],
                                                     'residue_policy': {'favored_residues': [],
                                                                        'disfavored_residues': [],
                                                                        'policy_weight': 1.0,
                                                                        'position_residue_rules': {}}},
                                                    {'node_id': 'llm_binder_tile_05',
                                                     'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                                  'chain_id': 'B',
                                                                  'spans': [[40, 50]]},
                                                     'action_profile': 'point_resample',
                                                     'intent': 'Explore this equal-length binder tile '
                                                               'under the fixed task objectives.',
                                                     'evidence_refs': ['asyn:B:40',
                                                                       'asyn:B:41',
                                                                       'asyn:B:42',
                                                                       'asyn:B:43',
                                                                       'asyn:B:44',
                                                                       'asyn:B:45',
                                                                       'asyn:B:46',
                                                                       'asyn:B:47',
                                                                       'asyn:B:48',
                                                                       'asyn:B:49'],
                                                     'residue_policy': {'favored_residues': [],
                                                                        'disfavored_residues': [],
                                                                        'policy_weight': 1.0,
                                                                        'position_residue_rules': {}}}],
                               'mapping_edges': [{'edge_id': 'llm_tile_01_hotspot_interface',
                                                  'functional_node_id': 'hotspot_interface',
                                                  'structural_node_id': 'llm_binder_tile_01',
                                                  'action_operator': 'point',
                                                  'evidence_refs': ['asyn:B:0',
                                                                    'asyn:B:1',
                                                                    'asyn:B:2',
                                                                    'asyn:B:3',
                                                                    'asyn:B:4',
                                                                    'asyn:B:5',
                                                                    'asyn:B:6',
                                                                    'asyn:B:7',
                                                                    'asyn:B:8',
                                                                    'asyn:B:9']},
                                                 {'edge_id': 'llm_tile_01_binder_holo_fold',
                                                  'functional_node_id': 'binder_holo_fold',
                                                  'structural_node_id': 'llm_binder_tile_01',
                                                  'action_operator': 'site_resample',
                                                  'evidence_refs': ['asyn:B:0',
                                                                    'asyn:B:1',
                                                                    'asyn:B:2',
                                                                    'asyn:B:3',
                                                                    'asyn:B:4',
                                                                    'asyn:B:5',
                                                                    'asyn:B:6',
                                                                    'asyn:B:7',
                                                                    'asyn:B:8',
                                                                    'asyn:B:9']},
                                                 {'edge_id': 'llm_tile_02_hotspot_interface',
                                                  'functional_node_id': 'hotspot_interface',
                                                  'structural_node_id': 'llm_binder_tile_02',
                                                  'action_operator': 'point',
                                                  'evidence_refs': ['asyn:B:10',
                                                                    'asyn:B:11',
                                                                    'asyn:B:12',
                                                                    'asyn:B:13',
                                                                    'asyn:B:14',
                                                                    'asyn:B:15',
                                                                    'asyn:B:16',
                                                                    'asyn:B:17',
                                                                    'asyn:B:18',
                                                                    'asyn:B:19']},
                                                 {'edge_id': 'llm_tile_02_binder_holo_fold',
                                                  'functional_node_id': 'binder_holo_fold',
                                                  'structural_node_id': 'llm_binder_tile_02',
                                                  'action_operator': 'site_resample',
                                                  'evidence_refs': ['asyn:B:10',
                                                                    'asyn:B:11',
                                                                    'asyn:B:12',
                                                                    'asyn:B:13',
                                                                    'asyn:B:14',
                                                                    'asyn:B:15',
                                                                    'asyn:B:16',
                                                                    'asyn:B:17',
                                                                    'asyn:B:18',
                                                                    'asyn:B:19']},
                                                 {'edge_id': 'llm_tile_03_hotspot_interface',
                                                  'functional_node_id': 'hotspot_interface',
                                                  'structural_node_id': 'llm_binder_tile_03',
                                                  'action_operator': 'point',
                                                  'evidence_refs': ['asyn:B:20',
                                                                    'asyn:B:21',
                                                                    'asyn:B:22',
                                                                    'asyn:B:23',
                                                                    'asyn:B:24',
                                                                    'asyn:B:25',
                                                                    'asyn:B:26',
                                                                    'asyn:B:27',
                                                                    'asyn:B:28',
                                                                    'asyn:B:29']},
                                                 {'edge_id': 'llm_tile_03_hotspot_localization',
                                                  'functional_node_id': 'hotspot_localization',
                                                  'structural_node_id': 'llm_binder_tile_03',
                                                  'action_operator': 'site_resample',
                                                  'evidence_refs': ['asyn:B:20',
                                                                    'asyn:B:21',
                                                                    'asyn:B:22',
                                                                    'asyn:B:23',
                                                                    'asyn:B:24',
                                                                    'asyn:B:25',
                                                                    'asyn:B:26',
                                                                    'asyn:B:27',
                                                                    'asyn:B:28',
                                                                    'asyn:B:29']},
                                                 {'edge_id': 'llm_tile_04_hotspot_interface',
                                                  'functional_node_id': 'hotspot_interface',
                                                  'structural_node_id': 'llm_binder_tile_04',
                                                  'action_operator': 'point',
                                                  'evidence_refs': ['asyn:B:30',
                                                                    'asyn:B:31',
                                                                    'asyn:B:32',
                                                                    'asyn:B:33',
                                                                    'asyn:B:34',
                                                                    'asyn:B:35',
                                                                    'asyn:B:36',
                                                                    'asyn:B:37',
                                                                    'asyn:B:38',
                                                                    'asyn:B:39']},
                                                 {'edge_id': 'llm_tile_04_binder_holo_fold',
                                                  'functional_node_id': 'binder_holo_fold',
                                                  'structural_node_id': 'llm_binder_tile_04',
                                                  'action_operator': 'site_resample',
                                                  'evidence_refs': ['asyn:B:30',
                                                                    'asyn:B:31',
                                                                    'asyn:B:32',
                                                                    'asyn:B:33',
                                                                    'asyn:B:34',
                                                                    'asyn:B:35',
                                                                    'asyn:B:36',
                                                                    'asyn:B:37',
                                                                    'asyn:B:38',
                                                                    'asyn:B:39']},
                                                 {'edge_id': 'llm_tile_05_developability',
                                                  'functional_node_id': 'developability',
                                                  'structural_node_id': 'llm_binder_tile_05',
                                                  'action_operator': 'point',
                                                  'evidence_refs': ['asyn:B:40',
                                                                    'asyn:B:41',
                                                                    'asyn:B:42',
                                                                    'asyn:B:43',
                                                                    'asyn:B:44',
                                                                    'asyn:B:45',
                                                                    'asyn:B:46',
                                                                    'asyn:B:47',
                                                                    'asyn:B:48',
                                                                    'asyn:B:49']},
                                                 {'edge_id': 'llm_tile_05_binder_holo_fold',
                                                  'functional_node_id': 'binder_holo_fold',
                                                  'structural_node_id': 'llm_binder_tile_05',
                                                  'action_operator': 'site_resample',
                                                  'evidence_refs': ['asyn:B:40',
                                                                    'asyn:B:41',
                                                                    'asyn:B:42',
                                                                    'asyn:B:43',
                                                                    'asyn:B:44',
                                                                    'asyn:B:45',
                                                                    'asyn:B:46',
                                                                    'asyn:B:47',
                                                                    'asyn:B:48',
                                                                    'asyn:B:49']}],
                               'decision_record': {'action': 'create',
                                                   'diagnosis': 'The task specifies a fixed 50-residue '
                                                                'binder and an immutable ASYN target.',
                                                   'hypothesis': 'Five equal tiles provide initial '
                                                                 'coverage for fold, interface, hotspot '
                                                                 'localization, and developability '
                                                                 'objectives.',
                                                   'evidence_refs': ['asyn:B:0',
                                                                     'asyn:B:1',
                                                                     'asyn:B:2',
                                                                     'asyn:B:3',
                                                                     'asyn:B:4',
                                                                     'asyn:B:5',
                                                                     'asyn:B:6',
                                                                     'asyn:B:7',
                                                                     'asyn:B:8',
                                                                     'asyn:B:9',
                                                                     'asyn:B:10',
                                                                     'asyn:B:11',
                                                                     'asyn:B:12',
                                                                     'asyn:B:13',
                                                                     'asyn:B:14',
                                                                     'asyn:B:15',
                                                                     'asyn:B:16',
                                                                     'asyn:B:17',
                                                                     'asyn:B:18',
                                                                     'asyn:B:19',
                                                                     'asyn:B:20',
                                                                     'asyn:B:21',
                                                                     'asyn:B:22',
                                                                     'asyn:B:23',
                                                                     'asyn:B:24',
                                                                     'asyn:B:25',
                                                                     'asyn:B:26',
                                                                     'asyn:B:27',
                                                                     'asyn:B:28',
                                                                     'asyn:B:29',
                                                                     'asyn:B:30',
                                                                     'asyn:B:31',
                                                                     'asyn:B:32',
                                                                     'asyn:B:33',
                                                                     'asyn:B:34',
                                                                     'asyn:B:35',
                                                                     'asyn:B:36',
                                                                     'asyn:B:37',
                                                                     'asyn:B:38',
                                                                     'asyn:B:39',
                                                                     'asyn:B:40',
                                                                     'asyn:B:41',
                                                                     'asyn:B:42',
                                                                     'asyn:B:43',
                                                                     'asyn:B:44',
                                                                     'asyn:B:45',
                                                                     'asyn:B:46',
                                                                     'asyn:B:47',
                                                                     'asyn:B:48',
                                                                     'asyn:B:49'],
                                                   'expected_effects': ['Expose all 50 binder positions '
                                                                        'under the declared editing '
                                                                        'contract.'],
                                                   'failure_condition': 'Reject any plan that violates '
                                                                        'coverage, target integrity, or '
                                                                        'the protected task constraints.',
                                                   'confidence': 0.0,
                                                   'rationale': 'Initial task definition with equal tile '
                                                                'weights and empty position-specific '
                                                                'residue rules.',
                                                   'rollback_condition': 'Retain the fixed starting '
                                                                         'sequence if no feasible '
                                                                         'candidate is found.'}}}
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
