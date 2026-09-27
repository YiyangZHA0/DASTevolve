"""Focused CPU regressions for released case wiring and bootstrap contracts."""
from copy import deepcopy
import csv
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
import yaml
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "outerloop")]

from cases.cab_lys3_hewl_cdr_recovery import cab_lys3_sequence_gate as gate
from outerloop.config import Config
from outerloop.structured_ast_proposal import extract_current_ast_revision_plan


class CaseConfigurationTests(unittest.TestCase):
    def test_aph_accepts_only_documented_llm_environment(self):
        env = {
            "ASTEVOLVE_LLM_API_BASE": "https://validation.invalid/v1",
            "ASTEVOLVE_LLM_API_KEY": "test-placeholder",
            "ASTEVOLVE_LLM_MODEL": "test-model",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_yaml(str(ROOT / "cases/aph3iia_kanamycin_active_site_preservation/config.yaml"))
        self.assertEqual(cfg.max_iterations, 50)
        self.assertEqual(cfg.llm.api_base, env["ASTEVOLVE_LLM_API_BASE"])
        self.assertEqual(cfg.llm.api_key, env["ASTEVOLVE_LLM_API_KEY"])

    def test_demo_entrypoint_registers_its_own_evaluator(self):
        env = {"ASTEVOLVE_PROJECT_ROOT": str(ROOT)}
        with patch.dict(os.environ, env, clear=True):
            module = importlib.import_module("cases.demo_case.initial_program")
            module = importlib.reload(module)
            preview = module.preview_case()
        self.assertEqual(preview["mask_true_counts"], {"B": 10, "T": 0})
        self.assertEqual(preview["fixed_residue_counts"], {"B": 2, "T": 6})


class FreshCaseInitializationTests(unittest.TestCase):
    def test_initial_programs_are_editable_without_preloaded_feedback_rules(self):
        for path in sorted((ROOT / "cases").glob("*/initial_program.py")):
            if path.parent.name == "demo_case":
                continue
            with self.subTest(case=path.parent.name):
                # Use the real outer-loop parser: generated/nonliteral plans
                # would compile locally but fail before the first LLM revision.
                plan = extract_current_ast_revision_plan(path.read_text(encoding="utf-8"))
                self.assertEqual(plan["decision_record"]["action"], "create")
                self.assertEqual(plan["decision_record"]["confidence"], 0.0)
                for node in plan["structural_nodes"]:
                    policy = node["residue_policy"]
                    self.assertEqual(policy.get("position_residue_rules", {}), {})
                    self.assertEqual(policy.get("policy_weight", 1.0), 1.0)

    def test_initial_adaptive_memory_contains_no_observations(self):
        def empty_tree(value):
            if isinstance(value, dict):
                return all(empty_tree(item) for item in value.values())
            if isinstance(value, list):
                return not value
            return value is None

        for path in sorted((ROOT / "cases").glob("*/memory.yaml")):
            with self.subTest(case=path.parent.name):
                memory = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertTrue(empty_tree(memory.get("adaptive_memory", {})))
                self.assertFalse(memory.get("entries"))

    def test_asyn_prompt_does_not_transfer_previous_search_observations(self):
        prompt = (ROOT / "cases/asyn_c_terminal_hotspot_binder/prompts/system_message.txt").read_text(encoding="utf-8")
        for leaked_context in (
            "CONTROLLER-SUPPLIED TRANSFERRED HELIX-BRIDGE PRIOR",
            "separate completed Full DAST continuation",
            "saved ESMFold/P-SEA analysis",
        ):
            self.assertNotIn(leaked_context, prompt)


class CABBootstrapContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = Path(self.temp.name)
        # Synthetic reference rows isolate scope logic; no archived data or model
        # weights are needed. The real sequence-complexity gate remains active.
        rows = [
            ("ref-1", "TGFDERN", "SYHQK", "YGWDSRNTVEKQHFLMAYGDSVT"),
            ("ref-2", "YGSDQKM", "TFNRE", "DGRYVQSTKHFWENLMAYGDTVS"),
            ("ref-3", "RNTYQEF", "KSDHW", "QYGTNVDRESWHKFLMAYGDTQS"),
        ]
        with (self.index / "embedding_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["reference_id", "cdr1", "cdr2", "cdr3"])
            writer.writerows(rows)
        self.active = sorted({i for values in gate.SABDAB_RULE_POSITIONS.values() for i in values})
        sequence = list("A" * 133)
        for i in (21, 32, 95, 108):
            sequence[i] = "C"
        self.root = {"A": "".join(sequence), "L": "L" * 129}
        self.fixed = {
            "A": {i: aa for i, aa in enumerate(self.root["A"]) if i not in self.active},
            "L": {i: aa for i, aa in enumerate(self.root["L"])},
        }
        self.masks = {"A": [i in self.active for i in range(133)], "L": [False] * 129}
        self.score = {"mutation_scope_contract": {
            "active_positions_by_chain": {"A": self.active, "L": []},
            "active_positions_by_node": {
                "current_" + region: {"chain_id": "A", "positions": list(positions)}
                for region, positions in gate.SABDAB_RULE_POSITIONS.items()
            },
            "max_total_mutations": 35,
        }}
        self.args = dict(root_sequences=self.root,
                         config={"sabdab_prior_enabled": True, "sabdab_prior_index_dir": str(self.index)},
                         count=2, seed=17, masks=self.masks,
                         fixed_residues=self.fixed, score_config=self.score)
        self.embedding = patch.object(gate, "score_sabdab_vhh_prior", return_value={"available": True, "plausibility": 0.8})
        self.embedding.start()
        self.addCleanup(self.embedding.stop)
        self.addCleanup(gate.sabdab_position_residue_rules.cache_clear)

    def test_bootstrap_preserves_fixed_chains_and_uses_current_node_owners(self):
        seeds = gate.build_sabdab_bootstrap_proposals(**self.args)
        self.assertEqual(len(seeds), 2)
        for sequences, move in seeds:
            changed = {i for i, (a, b) in enumerate(zip(self.root["A"], sequences["A"])) if a != b}
            self.assertTrue(changed.issubset(self.active))
            self.assertEqual(sequences["L"], self.root["L"])
            for i, aa in self.fixed["A"].items():
                self.assertEqual(sequences["A"][i], aa)
            owners = move["mapping_realization_summary"]["position_owners"]
            self.assertEqual({row["position"] for row in owners}, changed)
            self.assertTrue(all(row["owner_node_id"].startswith("current_") for row in owners))

    def test_rescoped_bootstrap_changes_only_new_mask_and_owner(self):
        parent = gate.build_sabdab_bootstrap_proposals(**self.args)[0][0]
        score = {"mutation_scope_contract": {
            "active_positions_by_chain": {"A": [25, 26], "L": []},
            "active_positions_by_node": {"new_cdr1": {"chain_id": "A", "positions": [25, 26]}},
            "max_total_mutations": 2,
        }}
        masks = {"A": [i in {25, 26} for i in range(133)], "L": [False] * 129}
        args = {**self.args, "root_sequences": parent, "masks": masks, "score_config": score}
        for sequences, move in gate.build_sabdab_bootstrap_proposals(**args):
            changed = {i for i, (a, b) in enumerate(zip(parent["A"], sequences["A"])) if a != b}
            self.assertTrue(changed)
            self.assertTrue(changed.issubset({25, 26}))
            self.assertEqual(sequences["L"], parent["L"])
            self.assertTrue(all(row["owner_node_id"] == "new_cdr1" for row in move["mapping_realization_summary"]["position_owners"]))

    def test_bootstrap_rejects_missing_or_inconsistent_compiled_scope(self):
        with self.assertRaisesRegex(ValueError, "requires compiled masks"):
            gate.build_sabdab_bootstrap_proposals(**{**self.args, "masks": None})
        score = deepcopy(self.score)
        score["mutation_scope_contract"]["active_positions_by_chain"]["A"] = [25]
        with self.assertRaisesRegex(ValueError, "scope disagree"):
            gate.build_sabdab_bootstrap_proposals(**{**self.args, "score_config": score})


if __name__ == "__main__":
    unittest.main()
