"""CPU regressions for structured proposals, configuration, and island policy."""
import asyncio
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'outerloop')]

from dacite.exceptions import UnexpectedDataError
from outerloop.config import Config, DatabaseConfig, LLMModelConfig
from outerloop.database import Program, ProgramDatabase
from outerloop.evolution_policy import EvolutionCandidate, decide_parent, decide_migration
from outerloop.structured_ast_proposal import (
    STRUCTURED_AST_AUDIT_VERSION, STRUCTURED_AST_CONTROLS,
    STRUCTURED_AST_PROPOSAL_VERSION, apply_structured_ast_proposal,
    extract_current_ast_revision_plan, parent_evolve_hash,
)
from outerloop.utils.code_utils import CandidateDiffError


def candidate(identifier, score, passed=True):
    return EvolutionCandidate.create(
        candidate_id=identifier, objective=score,
        gate_sources={'test_evaluator': {'passed': passed, 'reasons': [] if passed else ['protected_residue']}},
        phenotype_hash='phenotype:' + identifier,
        effective_contract_hash='contract:' + identifier,
        sequence_bundle_hash='sequence:' + identifier,
    )


class OuterReleaseTests(unittest.TestCase):
    def test_unknown_config_fields_fail_instead_of_using_defaults(self):
        for raw in [{'max_iteratons': 50}, {'database': {'num_isalnds': 4}},
                    {'llm': {'temperatur': 0.2}}]:
            with self.subTest(raw=raw), self.assertRaises(UnexpectedDataError):
                Config.from_dict(raw)

    def test_all_shipped_configs_load_strictly(self):
        for path in sorted((ROOT / 'cases').glob('*/config.yaml')):
            env = {key: 'fake' for key in re.findall(r'\$\{([A-Z0-9_]+)\}', path.read_text(encoding='utf-8'))}
            with self.subTest(case=path.parent.name), patch.dict(os.environ, env):
                config = Config.from_yaml(path)
                self.assertGreater(config.database.num_islands, 0)
                if path.parent.name.startswith(('asyn_', 'cab_', 'aph')):
                    self.assertEqual(config.max_iterations, 50)

    def test_cases_share_the_documented_llm_environment(self):
        env = {'ASTEVOLVE_LLM_API_BASE': 'https://example.invalid/v1',
               'ASTEVOLVE_LLM_API_KEY': 'test-only', 'ASTEVOLVE_LLM_MODEL': 'test-model'}
        with patch.dict(os.environ, env, clear=True):
            for path in sorted((ROOT / 'cases').glob('*/config.yaml')):
                with self.subTest(case=path.parent.name):
                    config = Config.from_yaml(path)
                    self.assertEqual(config.llm.models[0].api_key, 'test-only')

    def test_api_retry_budget_has_one_owner(self):
        from outerloop.llm.openai import OpenAILLM
        model = LLMModelConfig(name='test-model', api_key='test-only',
            api_base='https://example.invalid/v1', retries=2, retry_delay=0, timeout=1)
        with patch('outerloop.llm.openai.openai.OpenAI') as sdk:
            client = OpenAILLM(model)
            self.assertEqual(sdk.call_args.kwargs['max_retries'], 0)
        attempts = []
        async def fake_call(params):
            attempts.append(params)
            if len(attempts) < 3:
                raise RuntimeError('synthetic transient error')
            return 'validated response'
        with patch.object(client, '_call_api', fake_call):
            result = asyncio.run(client.generate_with_context('system', [{'role': 'user', 'content': 'test'}]))
        self.assertEqual(result, 'validated response')
        self.assertEqual(len(attempts), 3)

    def test_asyn_roles_rank_their_declared_evidence_differently(self):
        path = ROOT / 'cases/asyn_c_terminal_hotspot_binder/config.yaml'
        env = {key: 'fake' for key in re.findall(r'\$\{([A-Z0-9_]+)\}', path.read_text(encoding='utf-8'))}
        with patch.dict(os.environ, env):
            config = Config.from_yaml(path)
        db = ProgramDatabase(config.database)
        roles = config.database.island_roles
        interface = Program('interface', 'x', metrics={'asyn_interface_quality': .95, 'asyn_hotspot_localization': .55})
        fold = Program('fold', 'x', metrics={'asyn_binder_holo_confidence': .95, 'asyn_binder_holo_confidence_floor': .9})
        localization = Program('localization', 'x', metrics={'asyn_interface_quality': .5, 'asyn_hotspot_localization': .95})
        safety = Program('safety', 'x', metrics={'asyn_binder_developability': .95, 'asyn_binder_hydrophobic_guard': 1., 'asyn_interface_clash_guard': 1.})
        programs = [interface, fold, localization, safety]
        winners = [max(programs, key=lambda p: db._island_role_sampling_bonus(p, role)).id for role in roles]
        self.assertEqual(winners, ['interface', 'fold', 'localization', 'safety'])
        missing = Program('missing', 'x', metrics={})
        self.assertTrue(all(db._island_role_sampling_bonus(missing, role) == 0 for role in roles))
        self.assertTrue(db._v9_population_policy_enabled())
        # Synthetic evaluated candidates enter the existing sampler directly;
        # production admission/sealing is a separate integration boundary.
        for program in programs:
            program.metrics['combined_score'] = .5
            program.metadata['outer_evolution_candidate'] = candidate(program.id, .5).to_dict()
        db.programs = {program.id: program for program in programs}
        db.islands = [set(db.programs) for _ in roles]
        selected, _ = db.sample_from_island(0, num_inspirations=0)
        decision = selected.metadata['last_outer_parent_selection']
        scheduler = decision['details']['ast_scheduler']
        self.assertTrue(scheduler['applied'])
        self.assertEqual(scheduler['island_role_id'], 'interface_engagement')
        factors = scheduler['sampling_factors']
        self.assertGreater(factors['interface'], factors['fold'])

    def test_gates_precede_scores_in_parent_selection_and_migration(self):
        feasible = candidate('feasible', .4)
        failed = candidate('failed', 1000., False)
        for mode in ['best', 'weighted', 'uniform', 'mixture']:
            result = decide_parent([failed, feasible], base_seed=17, namespace='test', decision_index=0, mode=mode)
            self.assertEqual(result.selected_candidate_id, 'feasible')
        migration = decide_migration(failed, [feasible], capacity=1, base_seed=17,
            namespace='test-migration', decision_index=1,
            secondary_scores={'failed': 1., 'feasible': 0.}, secondary_objective_tolerance=2000.)
        self.assertEqual(migration.add_ids, ())
        better_for_target = candidate('specialist', .395)
        accepted = decide_migration(better_for_target, [feasible], capacity=1,
            base_seed=17, namespace='test-migration', decision_index=2,
            secondary_scores={'specialist': .9, 'feasible': .1},
            secondary_label='target-role', secondary_objective_tolerance=.02)
        self.assertEqual(accepted.add_ids, ('specialist',))
        self.assertFalse(accepted.details['copy_evaluation'])

    def test_per_island_migration_counter(self):
        db = ProgramDatabase(DatabaseConfig(num_islands=4, migration_interval=2,
            migration_interval_scope='per_island', outer_effective_phenotype_enabled=True,
            outer_population_policy_version='v9'))
        db.increment_island_generation(0)
        self.assertFalse(db.should_migrate())
        db.increment_island_generation(0)
        self.assertEqual(db._migration_due_source_islands(), [0])
        db._mark_migration_completed([0])
        self.assertFalse(db.should_migrate())
        db.increment_island_generation(1)
        db.increment_island_generation(1)
        self.assertEqual(db._migration_due_source_islands(), [1])

    def test_worker_retries_invalid_structured_response_before_evaluation(self):
        # Real worker control flow + real closed-schema parser + fake LLM/evaluator.
        import outerloop.process_parallel as worker
        from outerloop.prompt.sampler import PromptSampler
        parent_code = (ROOT / 'cases/asyn_c_terminal_hotspot_binder/initial_program.py').read_text(encoding='utf-8')
        plan = deepcopy(extract_current_ast_revision_plan(parent_code))
        plan['structural_nodes'][0]['residue_policy']['policy_weight'] += .05
        audit = {
            'schema_version': STRUCTURED_AST_AUDIT_VERSION,
            'rationale': 'Synthetic feedback identifies a local policy change.',
            'hypothesis': 'A stronger permitted residue prior improves the test objective.',
            'expected_effects': ['Increase the synthetic evaluator objective.'],
            'failure_condition': 'Any protected residue is edited.',
            'rollback_condition': 'Restore the parent if its feasible score is higher.',
            'edit_contract_response': None,
        }
        response = json.dumps({'schema_version': STRUCTURED_AST_PROPOSAL_VERSION,
            'parent_evolve_hash': parent_evolve_hash(parent_code),
            'controls': STRUCTURED_AST_CONTROLS, 'audit': audit, 'ast_revision_plan': plan})
        config = Config.from_dict({'max_code_length': 200000,
            'prompt': {'proposal_mode': 'structured_strategy_v1', 'hierarchical_audit_v2': False, 'candidate_diff_retries': 1},
            'database': {'num_islands': 4, 'outer_effective_phenotype_enabled': False, 'experiment_registry_enabled': False},
            'hierarchical_design': {'enabled': False, 'proposal_critic_enabled': False}})

        class FakeLLM:
            def __init__(self): self.calls = []
            async def generate_with_context(self, **kwargs):
                self.calls.append(kwargs)
                return 'not JSON' if len(self.calls) == 1 else response

        class FakeEvaluator:
            def __init__(self): self.validated = []; self.evaluated = []
            async def validate_candidate_program(self, code, **kwargs):
                compile(code, '<synthetic-case>', 'exec')
                self.validated.append(code)
            async def evaluate_program(self, code, program_id, **kwargs):
                self.evaluated.append(code)
                return {'combined_score': .7, 'hard_gate_pass': 1.}
            def get_pending_artifacts(self, program_id):
                return {'evaluator_report': {'test_only': True, 'diagnosis': 'synthetic policy improvement'}}

        fake_llm, fake_evaluator = FakeLLM(), FakeEvaluator()
        parent = Program('parent', parent_code, metrics={'combined_score': .4}, metadata={'island': 2})
        snapshot = {'programs': {'parent': parent.to_dict()}, 'artifacts': {'parent': {'failure_diagnosis': 'TEST_FEEDBACK_SENTINEL'}},
                    'islands': [[], [], ['parent'], []], 'sampling_island': 2,
                    'island_roles': {'2': {'role_id': 'topology_exploration', 'focus': 'TEST_ROLE_SENTINEL'}}}
        with patch.multiple(worker, create=True, _worker_config=config, _worker_llm_ensemble=fake_llm,
                            _worker_evaluator=fake_evaluator, _worker_prompt_sampler=PromptSampler(config.prompt)):
            result = worker._run_iteration_worker(1, snapshot, 'parent', [])
        self.assertIsNotNone(result.child_program_dict, result.error)
        self.assertEqual(len(fake_llm.calls), 2)
        self.assertEqual(len(fake_evaluator.evaluated), 1)
        self.assertEqual(result.target_island, 2)
        self.assertEqual(result.child_program_dict['parent_id'], 'parent')
        self.assertIn('structured_json_only', json.dumps(fake_llm.calls[1]))
        self.assertIn('TEST_FEEDBACK_SENTINEL', fake_llm.calls[0]['messages'][0]['content'])
        self.assertIn('TEST_ROLE_SENTINEL', fake_llm.calls[0]['messages'][0]['content'])
        changed = extract_current_ast_revision_plan(result.child_program_dict['code'])
        self.assertEqual(changed['structural_nodes'][0]['residue_policy']['policy_weight'], plan['structural_nodes'][0]['residue_policy']['policy_weight'])
        bad = json.loads(response); bad['parent_evolve_hash'] = '0' * 64
        with self.assertRaises(CandidateDiffError):
            apply_structured_ast_proposal(parent_code, json.dumps(bad))
        async def invalid_reply(**kwargs):
            return 'not JSON'
        failed_evaluator = FakeEvaluator()
        with patch.multiple(worker, create=True, _worker_config=config,
                            _worker_llm_ensemble=fake_llm, _worker_evaluator=failed_evaluator,
                            _worker_prompt_sampler=PromptSampler(config.prompt)), \
             patch.object(fake_llm, 'generate_with_context', invalid_reply):
            rejected = worker._run_iteration_worker(2, snapshot, 'parent', [])
        self.assertIsNone(rejected.child_program_dict)
        self.assertTrue(rejected.error)
        self.assertEqual(failed_evaluator.evaluated, [])


if __name__ == '__main__':
    unittest.main()
