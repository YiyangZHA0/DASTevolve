"""Executable Dual-AST strategy for APH(3')-IIa kanamycin preservation."""

# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Any, Dict

from engine.default_strategy import base_strategy


def propose_strategy() -> Dict[str, Any]:
    """Return the evidence-grounded initial editable AST for APH(3')-IIa.

    The seed Nodes below are only the starting topology.  On every
    revision the LLM may keep, create, delete, migrate, resize, split, merge, or
    rewire 4--12 Nodes over 8--32 positions from the catalog migration frontier.
    Omitting an old Node deletes it; changing its selector migrates/resizes it.
    A Node may contain up to eight positions and four spans, but must remain
    inside one compiled segment.  Every selected position must cite its
    aph3iia:A:<zero-based-index> catalog record.

    The LLM also controls soft favored/disfavored amino-acid guidance at those
    positions.  The launch-selected case tier remains authoritative: Level-1
    and Level-2 preserve the five-site baselines, while frontier supplies a
    conservative per-position hard alphabet across the geometry-screened
    surface. Initial Nodes follow compiled-segment boundaries. The donor-context
    hold, verified D157/D190/N195/D208 core, KAN contact shell, and unresolved
    N terminus are never LLM writable.
    """

    strategy = base_strategy()
    strategy.update(
        {'preferred_edit_order': [],
 'outer_loop_phase': 'explore_ast',
 'ast_revision_plan': {'schema_version': 'astevolve.ast_revision_plan.v2',
                       'structural_nodes': [{'node_id': 'llm_site14',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[13, 14]]},
                                             'action_profile': 'conservative_point',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S03_EDIT_SITE14 for constrained sequence exploration '
                                                       'and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:13'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_nlobe_s04_surface_network',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[15, 17], [19, 20], [30, 34]]},
                                             'action_profile': 'exploratory_point_resample',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S04_NLOBE_PRE_DONOR for constrained sequence '
                                                       'exploration and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:15',
                                                               'aph3iia:A:16',
                                                               'aph3iia:A:19',
                                                               'aph3iia:A:30',
                                                               'aph3iia:A:31',
                                                               'aph3iia:A:32',
                                                               'aph3iia:A:33'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_site72',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[71, 72]]},
                                             'action_profile': 'conservative_point',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S07_EDIT_SITE72 for constrained sequence exploration '
                                                       'and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:71'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_nlobe_support',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[73, 74], [84, 85], [86, 88]]},
                                             'action_profile': 'exploratory_point_resample',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S08_NLOBE_REMAINDER for constrained sequence '
                                                       'exploration and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:73',
                                                               'aph3iia:A:84',
                                                               'aph3iia:A:86',
                                                               'aph3iia:A:87'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_clobe_hinge_support',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[124, 125], [126, 127], [128, 130]]},
                                             'action_profile': 'exploratory_point_resample',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S11_CLOBE_PRE_KAN_SHELL for constrained sequence '
                                                       'exploration and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:124',
                                                               'aph3iia:A:126',
                                                               'aph3iia:A:128',
                                                               'aph3iia:A:129'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_clobe_support',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[163, 164], [171, 172], [174, 176]]},
                                             'action_profile': 'exploratory_point_resample',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S13_CLOBE_PRE_SITE178 for constrained sequence '
                                                       'exploration and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:163',
                                                               'aph3iia:A:171',
                                                               'aph3iia:A:174',
                                                               'aph3iia:A:175'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_site178',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[177, 178]]},
                                             'action_profile': 'conservative_point',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S14_EDIT_SITE178 for constrained sequence '
                                                       'exploration and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:177'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_site238',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[237, 238]]},
                                             'action_profile': 'conservative_point',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S28_EDIT_SITE238 for constrained sequence '
                                                       'exploration and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:237'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}},
                                            {'node_id': 'llm_cterm_support',
                                             'selector': {'schema_version': 'astevolve.residue_selector.v1',
                                                          'chain_id': 'A',
                                                          'spans': [[238, 239], [251, 252], [254, 255]]},
                                             'action_profile': 'balanced_point_resample',
                                             'intent': 'Initialize catalog-eligible positions within '
                                                       'S29_CLOBE_PRE_CTERM_SHELL for constrained sequence '
                                                       'exploration and structural-preservation evaluation.',
                                             'evidence_refs': ['aph3iia:A:238',
                                                               'aph3iia:A:251',
                                                               'aph3iia:A:254'],
                                             'residue_policy': {'favored_residues': [],
                                                                'disfavored_residues': [],
                                                                'position_residue_rules': {},
                                                                'policy_weight': 1.0}}],
                       'mapping_edges': [{'edge_id': 'llm_site14_point',
                                          'functional_node_id': 'sequence_integrity',
                                          'structural_node_id': 'llm_site14',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:13']},
                                         {'edge_id': 'llm_site14_site_resample',
                                          'functional_node_id': 'mutation_scope_integrity',
                                          'structural_node_id': 'llm_site14',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:13']},
                                         {'edge_id': 'llm_site14_segment_resample',
                                          'functional_node_id': 'preserve_global_fold',
                                          'structural_node_id': 'llm_site14',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:13']},
                                         {'edge_id': 'llm_site14_segment_mutagenesis',
                                          'functional_node_id': 'preserve_verified_core',
                                          'structural_node_id': 'llm_site14',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:13']},
                                         {'edge_id': 'llm_nlobe_s04_surface_network_point',
                                          'functional_node_id': 'preserve_kanamycin_contacts',
                                          'structural_node_id': 'llm_nlobe_s04_surface_network',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:15',
                                                            'aph3iia:A:16',
                                                            'aph3iia:A:19',
                                                            'aph3iia:A:30',
                                                            'aph3iia:A:31',
                                                            'aph3iia:A:32',
                                                            'aph3iia:A:33']},
                                         {'edge_id': 'llm_nlobe_s04_surface_network_site_resample',
                                          'functional_node_id': 'preserve_kanamycin_pose',
                                          'structural_node_id': 'llm_nlobe_s04_surface_network',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:15',
                                                            'aph3iia:A:16',
                                                            'aph3iia:A:19',
                                                            'aph3iia:A:30',
                                                            'aph3iia:A:31',
                                                            'aph3iia:A:32',
                                                            'aph3iia:A:33']},
                                         {'edge_id': 'llm_nlobe_s04_surface_network_segment_resample',
                                          'functional_node_id': 'preserve_mg_geometry',
                                          'structural_node_id': 'llm_nlobe_s04_surface_network',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:15',
                                                            'aph3iia:A:16',
                                                            'aph3iia:A:19',
                                                            'aph3iia:A:30',
                                                            'aph3iia:A:31',
                                                            'aph3iia:A:32',
                                                            'aph3iia:A:33']},
                                         {'edge_id': 'llm_nlobe_s04_surface_network_segment_mutagenesis',
                                          'functional_node_id': 'clash_free',
                                          'structural_node_id': 'llm_nlobe_s04_surface_network',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:15',
                                                            'aph3iia:A:16',
                                                            'aph3iia:A:19',
                                                            'aph3iia:A:30',
                                                            'aph3iia:A:31',
                                                            'aph3iia:A:32',
                                                            'aph3iia:A:33']},
                                         {'edge_id': 'llm_site72_point',
                                          'functional_node_id': 'kanamycin_label_compatibility',
                                          'structural_node_id': 'llm_site72',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:71']},
                                         {'edge_id': 'llm_site72_site_resample',
                                          'functional_node_id': 'sequence_plausibility',
                                          'structural_node_id': 'llm_site72',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:71']},
                                         {'edge_id': 'llm_site72_segment_resample',
                                          'functional_node_id': 'repeat_robustness',
                                          'structural_node_id': 'llm_site72',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:71']},
                                         {'edge_id': 'llm_site72_segment_mutagenesis',
                                          'functional_node_id': 'sequence_diversity',
                                          'structural_node_id': 'llm_site72',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:71']},
                                         {'edge_id': 'llm_nlobe_support_point',
                                          'functional_node_id': 'sequence_integrity',
                                          'structural_node_id': 'llm_nlobe_support',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:73',
                                                            'aph3iia:A:84',
                                                            'aph3iia:A:86',
                                                            'aph3iia:A:87']},
                                         {'edge_id': 'llm_nlobe_support_site_resample',
                                          'functional_node_id': 'mutation_scope_integrity',
                                          'structural_node_id': 'llm_nlobe_support',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:73',
                                                            'aph3iia:A:84',
                                                            'aph3iia:A:86',
                                                            'aph3iia:A:87']},
                                         {'edge_id': 'llm_nlobe_support_segment_resample',
                                          'functional_node_id': 'preserve_global_fold',
                                          'structural_node_id': 'llm_nlobe_support',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:73',
                                                            'aph3iia:A:84',
                                                            'aph3iia:A:86',
                                                            'aph3iia:A:87']},
                                         {'edge_id': 'llm_nlobe_support_segment_mutagenesis',
                                          'functional_node_id': 'preserve_verified_core',
                                          'structural_node_id': 'llm_nlobe_support',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:73',
                                                            'aph3iia:A:84',
                                                            'aph3iia:A:86',
                                                            'aph3iia:A:87']},
                                         {'edge_id': 'llm_clobe_hinge_support_point',
                                          'functional_node_id': 'preserve_kanamycin_contacts',
                                          'structural_node_id': 'llm_clobe_hinge_support',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:124',
                                                            'aph3iia:A:126',
                                                            'aph3iia:A:128',
                                                            'aph3iia:A:129']},
                                         {'edge_id': 'llm_clobe_hinge_support_site_resample',
                                          'functional_node_id': 'preserve_kanamycin_pose',
                                          'structural_node_id': 'llm_clobe_hinge_support',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:124',
                                                            'aph3iia:A:126',
                                                            'aph3iia:A:128',
                                                            'aph3iia:A:129']},
                                         {'edge_id': 'llm_clobe_hinge_support_segment_resample',
                                          'functional_node_id': 'preserve_mg_geometry',
                                          'structural_node_id': 'llm_clobe_hinge_support',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:124',
                                                            'aph3iia:A:126',
                                                            'aph3iia:A:128',
                                                            'aph3iia:A:129']},
                                         {'edge_id': 'llm_clobe_hinge_support_segment_mutagenesis',
                                          'functional_node_id': 'clash_free',
                                          'structural_node_id': 'llm_clobe_hinge_support',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:124',
                                                            'aph3iia:A:126',
                                                            'aph3iia:A:128',
                                                            'aph3iia:A:129']},
                                         {'edge_id': 'llm_clobe_support_point',
                                          'functional_node_id': 'kanamycin_label_compatibility',
                                          'structural_node_id': 'llm_clobe_support',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:163',
                                                            'aph3iia:A:171',
                                                            'aph3iia:A:174',
                                                            'aph3iia:A:175']},
                                         {'edge_id': 'llm_clobe_support_site_resample',
                                          'functional_node_id': 'sequence_plausibility',
                                          'structural_node_id': 'llm_clobe_support',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:163',
                                                            'aph3iia:A:171',
                                                            'aph3iia:A:174',
                                                            'aph3iia:A:175']},
                                         {'edge_id': 'llm_clobe_support_segment_resample',
                                          'functional_node_id': 'repeat_robustness',
                                          'structural_node_id': 'llm_clobe_support',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:163',
                                                            'aph3iia:A:171',
                                                            'aph3iia:A:174',
                                                            'aph3iia:A:175']},
                                         {'edge_id': 'llm_clobe_support_segment_mutagenesis',
                                          'functional_node_id': 'sequence_diversity',
                                          'structural_node_id': 'llm_clobe_support',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:163',
                                                            'aph3iia:A:171',
                                                            'aph3iia:A:174',
                                                            'aph3iia:A:175']},
                                         {'edge_id': 'llm_site178_point',
                                          'functional_node_id': 'sequence_integrity',
                                          'structural_node_id': 'llm_site178',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:177']},
                                         {'edge_id': 'llm_site178_site_resample',
                                          'functional_node_id': 'mutation_scope_integrity',
                                          'structural_node_id': 'llm_site178',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:177']},
                                         {'edge_id': 'llm_site178_segment_resample',
                                          'functional_node_id': 'preserve_global_fold',
                                          'structural_node_id': 'llm_site178',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:177']},
                                         {'edge_id': 'llm_site178_segment_mutagenesis',
                                          'functional_node_id': 'preserve_verified_core',
                                          'structural_node_id': 'llm_site178',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:177']},
                                         {'edge_id': 'llm_site238_point',
                                          'functional_node_id': 'preserve_kanamycin_contacts',
                                          'structural_node_id': 'llm_site238',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:237']},
                                         {'edge_id': 'llm_site238_site_resample',
                                          'functional_node_id': 'preserve_kanamycin_pose',
                                          'structural_node_id': 'llm_site238',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:237']},
                                         {'edge_id': 'llm_site238_segment_resample',
                                          'functional_node_id': 'preserve_mg_geometry',
                                          'structural_node_id': 'llm_site238',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:237']},
                                         {'edge_id': 'llm_site238_segment_mutagenesis',
                                          'functional_node_id': 'clash_free',
                                          'structural_node_id': 'llm_site238',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:237']},
                                         {'edge_id': 'llm_cterm_support_point',
                                          'functional_node_id': 'kanamycin_label_compatibility',
                                          'structural_node_id': 'llm_cterm_support',
                                          'action_operator': 'point',
                                          'evidence_refs': ['aph3iia:A:238',
                                                            'aph3iia:A:251',
                                                            'aph3iia:A:254']},
                                         {'edge_id': 'llm_cterm_support_site_resample',
                                          'functional_node_id': 'sequence_plausibility',
                                          'structural_node_id': 'llm_cterm_support',
                                          'action_operator': 'site_resample',
                                          'evidence_refs': ['aph3iia:A:238',
                                                            'aph3iia:A:251',
                                                            'aph3iia:A:254']},
                                         {'edge_id': 'llm_cterm_support_segment_resample',
                                          'functional_node_id': 'repeat_robustness',
                                          'structural_node_id': 'llm_cterm_support',
                                          'action_operator': 'segment_resample',
                                          'evidence_refs': ['aph3iia:A:238',
                                                            'aph3iia:A:251',
                                                            'aph3iia:A:254']},
                                         {'edge_id': 'llm_cterm_support_segment_mutagenesis',
                                          'functional_node_id': 'sequence_diversity',
                                          'structural_node_id': 'llm_cterm_support',
                                          'action_operator': 'segment_mutagenesis',
                                          'evidence_refs': ['aph3iia:A:238',
                                                            'aph3iia:A:251',
                                                            'aph3iia:A:254']}],
                       'decision_record': {'action': 'create',
                                           'diagnosis': 'Initialize fixed-length APH preservation design '
                                                        'from the P00552 wild-type sequence and static 1ND4 '
                                                        'residue evidence.',
                                           'hypothesis': 'Use nine segment-bounded structural nodes over 26 '
                                                         'catalog-eligible positions. Distribute the twelve '
                                                         'functional evaluation responsibilities in a fixed, '
                                                         'balanced order across legal actions as initial '
                                                         'hypotheses.',
                                           'evidence_refs': ['aph3iia:A:13',
                                                             'aph3iia:A:15',
                                                             'aph3iia:A:16',
                                                             'aph3iia:A:19',
                                                             'aph3iia:A:30',
                                                             'aph3iia:A:31',
                                                             'aph3iia:A:32',
                                                             'aph3iia:A:33',
                                                             'aph3iia:A:71',
                                                             'aph3iia:A:73',
                                                             'aph3iia:A:84',
                                                             'aph3iia:A:86',
                                                             'aph3iia:A:87',
                                                             'aph3iia:A:124',
                                                             'aph3iia:A:126',
                                                             'aph3iia:A:128',
                                                             'aph3iia:A:129',
                                                             'aph3iia:A:163',
                                                             'aph3iia:A:171',
                                                             'aph3iia:A:174',
                                                             'aph3iia:A:175',
                                                             'aph3iia:A:177',
                                                             'aph3iia:A:237',
                                                             'aph3iia:A:238',
                                                             'aph3iia:A:251',
                                                             'aph3iia:A:254'],
                                           'expected_effects': ['Define initial editable regions inside the '
                                                                'catalog and protected-span boundaries.',
                                                                'Preserve all twelve evaluator objectives '
                                                                'and the case-owned hard amino-acid '
                                                                'contract.',
                                                                'Permit subsequent revisions using evidence '
                                                                'generated during the current run.'],
                                           'failure_condition': 'Reject revisions outside catalog '
                                                                'eligibility, protected-span, segment, or '
                                                                'hard-alphabet limits.',
                                           'confidence': 0.0}}}
    )
    return strategy


# EVOLVE-BLOCK-END

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

from astevolve.runtime.case_context import current_case_kwargs
from astevolve.runtime.case_program import merge_locked_runtime_defaults
from astevolve.runtime.paths import artifact_path, data_path
from engine.case_builder import prepare_case_inputs, run_design_search


# One outer trial owns one effective-contract identity. Multi-seed robustness
# belongs to the explicit provider states/evaluator; repeating this call would
# correctly collide with the experiment registry's duplicate-contract barrier.
OUTER_EVALUATION_TRIALS = 1
_CASE_MANIFEST_OVERRIDE = str(os.environ.get("ASTEVOLVE_CASE_MANIFEST") or "").strip()
CASE_ROOT = (
    Path(_CASE_MANIFEST_OVERRIDE).expanduser().resolve().parent
    if _CASE_MANIFEST_OVERRIDE
    else Path(__file__).resolve().parent
)
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

from aph3iia_evaluator import register_aph3iia_plugin


register_aph3iia_plugin()

_WT_SEQUENCE = (
    "MIEQDGLHAGSPAAWVERLFGYDWAQQTIGCSDAAVFRLSAQGRPVLFVKTDLSGALNEL"
    "QDEAARLSWLATTGVPCAAVLDVVTEAGRDWLLLGEVPGQDLLSSHLAPAEKVSIMADAM"
    "RRLHTLDPATCPFDHQAKHRIERARTRMEAGLVDQDDLDEEHQGLAPAELFARLKARMPD"
    "GEDLVVTHGDACLPNIMVENGRFSGFIDCGRLGVADRYQDIALATRDIAEELGGEWADRF"
    "LVLYGIAAPDSQRIAFYRLLDEFF"
)


def _active_residue_tier() -> str:
    tier = str(os.environ.get("ASTEVOLVE_APH_DESIGN_TIER") or "frontier").strip()
    if tier not in {"level1", "level2", "frontier"}:
        raise ValueError(
            "ASTEVOLVE_APH_DESIGN_TIER must be level1, level2, or frontier"
        )
    return tier


_ACTIVE_RESIDUE_TIER = _active_residue_tier()

_LOCKED_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "resume_template_seqs": {"A": _WT_SEQUENCE},
    "case_owned_residue_policy_tier": _ACTIVE_RESIDUE_TIER,
    "iterations": 100,
    "search_method": "mcts",
    "mutation_ops": {
        "point": 0.35,
        "site_resample": 0.25,
        "segment_resample": 0.20,
        "segment_mutagenesis": 0.20,
    },
    "max_total_mutations": 8,
    "exploit_max_mutations": 4,
    "explore_max_mutations": 8,
    "repair_max_mutations": 2,
    "proposal_tier_mode": "mixed",
    "proposal_exploit_frac": 0.50,
    "proposal_explore_frac": 0.40,
    "proposal_repair_frac": 0.10,
    "search_schedule": {
        "enabled": True,
        "hard_max_total_mutations": 8,
        "phase_policy": (
            "Phase/posture may change Node topology, active positions, and soft residue "
            "guidance inside the launch-selected case tier. Frontier mode may migrate "
            "within catalog eligibility; protected spans and hard alphabets remain fixed."
        ),
        "phases": {
            "explore_ast": {
                "proposal_exploit_frac": 0.45,
                "proposal_explore_frac": 0.45,
                "proposal_repair_frac": 0.10,
            },
            "refine_ast": {
                "proposal_exploit_frac": 0.65,
                "proposal_explore_frac": 0.22,
                "proposal_repair_frac": 0.13,
            },
            "converge": {
                "proposal_exploit_frac": 0.78,
                "proposal_explore_frac": 0.08,
                "proposal_repair_frac": 0.14,
            },
        },
    },
    "fast_filter_enabled": True,
    "progen_weight": 0.05,
    "progen_chains": ["A"],
    "sequence_prior_model": "progen",
    "mcts_iteration_unit": "evaluated_unique_candidates",
    "mcts_candidate_budget_max_round_multiplier": 4,
    "mcts_candidate_budget_fail_on_underfill": True,
    "inner_structure_enabled": True,
    "inner_structure_model": 'protenix',
    "inner_structure_model_name": 'protenix_mini_esm_v0.5.0',
    "inner_structure_weight": 1.0,
    "inner_structure_fail_closed": True,
    "inner_structure_hard_gate": True,
    "sequence_generator_id": "deterministic_constraint_aware_v1",
    "sequence_generator_structure_condition_refs": [
        "candidate_chainA_KAN_MG"
    ],
    "sequence_generator_state_condition_refs": [
        "candidate_apo",
        "candidate_chainA_KAN_MG",
    ],
    "node_optimizer_enabled": True,
    "node_optimizer_candidate_count": 8,
    "node_optimizer_beam_width": 16,
    "node_optimizer_top_k_per_position": 4,
    "node_optimizer_temperature": 0.80,
    "node_optimizer_diversity_weight": 0.20,
    "node_optimizer_mutation_penalty": 0.12,
    "node_optimizer_prior_model": "masked_lm",
    "semantic_required_nodes": [],
    "semantic_anchor_nodes": [],
    "semantic_coverage_mode": "soft",
    "chai1_enabled": True,
    "chai1_top_frac": 1.0,
    "chai1_min_candidates": 8,
    "chai1_max_candidates": 8,
    "structure_model": "protenix",
    "structure_model_name": "protenix_mini_esm_v0.5.0",
    "protenix_model_name": "protenix_mini_esm_v0.5.0",
    "protenix_conda_env": "protenix",
    "protenix_seed": 101,
    "protenix_complex_use_msa": False,
    "protenix_complex_cycle": 1,
    "protenix_complex_step": 20,
    "protenix_complex_sample": 1,
    "protenix_complex_use_default_params": False,
    "protenix_complex_timeout": 1800,
    "structure_batch_size": 0,
    "structure_parallel_workers": 1,
    "structure_shortlist_policy": "formal_layered_novel",
    "structure_screen_single_node_diagnostic_quota": 0,
    "structure_selection_objective": "outer_aligned",
    "structure_allow_low_fidelity_fallback": False,
    "structure_screen_enabled": False,
    "structure_prescreen_enabled": False,
    "structure_prescreen_model": "esmfold2",
    "structure_prescreen_model_name": "biohub/ESMFold2",
    "structure_prescreen_top_frac": 1.0,
    "structure_prescreen_min_candidates": 10,
    "structure_prescreen_max_candidates": 10,
    "structure_prescreen_forward_all_to_screen": False,
    "structure_screen_all_candidates": False,
    "structure_screen_model": "protenix",
    "structure_screen_model_name": "protenix_mini_esm_v0.5.0",
    "structure_screen_top_frac": 1.0,
    "structure_screen_min_candidates": 10,
    "structure_screen_max_candidates": 10,
    "structure_rerank_enabled": False,
    "structure_rerank_model": "alphafold3",
    "structure_rerank_top_frac": 0.34,
    "structure_rerank_min_candidates": 4,
    "structure_rerank_max_candidates": 4,
    "structure_rerank_all_infeasible_rescue": True,
    "structure_physics_max_candidates": 0,
    "multistate_objectives_enabled": False,
    "multistate_objective_weight": 0.0,
    "mcts_save_tree": True,
    "mcts_save_variants": True,
    "mcts_artifact_mode": "normalized",
    "executable_island_policy_enabled": True,
    "score_config": {
        "weight_fast": 0.20,
        "weight_plddt": 0.15,
        "weight_iptm": 0.0,
        "weight_ptm": 0.05,
        "weight_ranking_score": 0.0,
        "weight_interface_plddt": 0.0,
        "weight_node_plddt_min": 0.10,
        "weight_clash": 0.10,
        "weight_multistate": 0.0,
        "weight_evaluator": 6.0,
        "inner_evaluator_loss_weight": 6.0,
        "inner_hard_gate_fail_penalty": 1000.0,
        "eval_primary_engagement": 1.5,
        "eval_pyrosetta": 0.0,
        "evaluator_backends": {
            "pyrosetta": {
                "enabled": True,
                "required": False,
                "timeout": 300,
                "analysis_role": "clash_and_total_energy_sanity",
                "defer_until_stage": "physics",
            }
        },
        "plddt_scale": 100.0,
        "clash_scale": 10.0,
        "fast_loss_nonneg": True,
        "evaluator_plugins": ["aph3iia_kanamycin"],
        "plugin_config": {
            "aph3iia_kanamycin": {
                "evaluation_mode": "candidate_complex_required",
                "reference_chain_a_path": str(
                    data_path("aph3iia_kanamycin_active_site_preservation", "structures/ast_ready/1ND4_chainA_KAN_MG.pdb")
                ),
                "reference_chain_b_path": str(
                    data_path("aph3iia_kanamycin_active_site_preservation", "structures/ast_ready/1ND4_chainB_KAN_MG.pdb")
                ),
                "resolved_scientific_span_1based": [10, 264],
                "verified_core_positions_1based": [157, 190, 195, 208],
                "kanamycin_contact_positions_1based": [
                    157, 158, 159, 160, 190, 195, 211,
                    226, 227, 230, 261, 262, 264,
                ],
                "global_backbone_rmsd_max": 1.5,
                "core_backbone_rmsd_max": 0.75,
                "core_all_heavy_rmsd_max": 1.25,
                "kanamycin_contact_cutoff": 4.5,
                "kanamycin_contact_coverage_min": 0.90,
                "kanamycin_pose_rmsd_max": 1.5,
                "mg_distance_error_max": 0.5,
                "severe_clash_count_max": 0,
                "require_generated_candidate_ligand": True,
                "allow_reference_heteroatom_transplant": False,
                "repeat_state_required": False,
                "evaluator_weights": {
                    "aph_sequence_integrity": 2.0,
                    "aph_mutation_scope_integrity": 2.0,
                    "aph_global_fold_preservation": 3.0,
                    "aph_verified_core_preservation": 4.0,
                    "aph_kanamycin_contact_preservation": 4.0,
                    "aph_kanamycin_pose_preservation": 3.0,
                    "aph_mg_geometry_preservation": 3.0,
                    "aph_clash_free": 3.0,
                    "aph_kanamycin_label_compatibility": 1.5,
                    "aph_sequence_plausibility": 1.0,
                    "aph_repeat_robustness": 1.0,
                    "aph_sequence_diversity": 0.5,
                },
            }
        },
    },
}


def runtime_strategy() -> Dict[str, Any]:
    """Merge the evolvable AST with controller-owned scientific controls."""

    return merge_locked_runtime_defaults(
        propose_strategy(),
        _LOCKED_RUNTIME_DEFAULTS,
        base_strategy,
    )


def _case_paths() -> Dict[str, str]:
    if os.environ.get("ASTEVOLVE_CASE_MANIFEST"):
        return current_case_kwargs()
    return {
        "design_state_path": str(CASE_ROOT / "design_state.json"),
        "memory_path": str(CASE_ROOT / "memory.yaml"),
    }


def preview_case() -> Dict[str, Any]:
    """Compile and expose the effective AST, tier, masks, and evaluator wiring."""

    prepared = prepare_case_inputs(runtime_strategy(), **_case_paths())
    compiled = prepared.blueprint.compile()
    return {
        "task_name": prepared.design_state["task_name"],
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
        "mask_true_counts": {
            chain_id: int(sum(mask))
            for chain_id, mask in prepared.masks.items()
        },
        "fixed_residue_counts": {
            chain_id: len(residues)
            for chain_id, residues in prepared.fixed_residues.items()
        },
        "active_residue_tier": _ACTIVE_RESIDUE_TIER,
        "residue_mutation_contract": {
            chain_id: {
                str(position): residues
                for position, residues in sorted(position_rules.items())
            }
            for chain_id, position_rules in sorted(
                prepared.resolved_strategy.get(
                    "residue_mutation_contract", {}
                ).items()
            )
        },
        "node_policy_names": sorted(
            prepared.resolved_strategy.get("node_edit_policies", {})
        ),
        "executable_structural_nodes": [
            node.to_dict()
            for node in prepared.executable_node_plan.structural_nodes
        ],
        "measurement_intents": [
            intent.to_dict()
            for intent in prepared.executable_node_plan.measurement_intents
        ],
        "effective_mapping_schedule": (
            prepared.effective_mapping_schedule.to_dict()
        ),
        "ast_revision_report": prepared.design_state.get(
            "_ast_revision_report", {}
        ),
        "case_owned_residue_policy_resolution": prepared.resolved_strategy.get(
            "case_owned_residue_policy_resolution", {}
        ),
        "mutation_scope_contract": prepared.score_config.get(
            "mutation_scope_contract", {}
        ),
        "evaluator_plugin_resolution": prepared.design_state.get(
            "_evaluator_plugin_resolution", {}
        ),
    }


def run_search(seed: Optional[int] = None) -> Dict[str, Any]:
    """Run one inner search with deferred memory commit and durable artifacts."""

    strategy = deepcopy(runtime_strategy())
    evaluation_id = (
        f"{Path(__file__).stem}__{_ACTIVE_RESIDUE_TIER}__seed_{int(seed or 0)}"
    )
    strategy["mcts_output_dir"] = str(
        artifact_path(
            "aph3iia_kanamycin_active_site_preservation",
            "inner",
            evaluation_id,
        )
    )
    return run_design_search(
        strategy,
        seed=seed,
        memory_commit_mode="deferred",
        **_case_paths(),
    )


__all__ = [
    "OUTER_EVALUATION_TRIALS",
    "preview_case",
    "propose_strategy",
    "run_search",
    "runtime_strategy",
]
