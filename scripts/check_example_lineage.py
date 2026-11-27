"""Regression checks for accepted rounds in the packaged examples. No model calls."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "outerloop")]

from astevolve.cases.example_lineage import verified_stage
from engine.experiment_identity import SequenceBundleIdentity


class VerifiedRoundTest(unittest.TestCase):
    def setUp(self):
        seqs = {"A": "ACDE"}
        winner = SequenceBundleIdentity.create(seqs).to_dict()
        self.parent = {"program_id": "p", "winner_sequence_hash": winner["sequence_bundle_hash"]}
        self.program = SimpleNamespace(
            id="c", parent_id="p", iteration_found=2, metrics={"hard_gate_pass": False},
            metadata={"accepted_runtime_artifact": {
                "acceptance": {"status": "accepted"}, "sequence_bundle_identity": winner,
            }},
        )
        self.search = {
            "root_candidate": {"seqs": seqs},
            "expansion_round_accounting": {
                "iteration_unit": "expansion_rounds", "requested_rounds": 10, "physical_rounds": 10,
            },
            "mcts_candidate_budget": {"completed_evaluated_unique_candidates": 6},
            "inner_structure_summary": {"evaluated_count": 7, "failure_count": 0},
            "mcts_tree_quality": {"required": True, "pass": True},
        }

    def verify(self):
        return verified_stage(self.program, {"search_artifacts": json.dumps(self.search)}, self.parent, 10)

    def test_completed_search_can_retain_an_infeasible_sequence(self):
        self.assertFalse(self.verify()["hard_gate_pass"])

    def test_matching_program_id_with_wrong_root_is_rejected(self):
        self.search["root_candidate"]["seqs"] = {"A": "AAAA"}
        with self.assertRaisesRegex(ValueError, "inherit"):
            self.verify()

    def test_physical_rounds_with_all_failed_structures_are_rejected(self):
        self.search["inner_structure_summary"]["failure_count"] = 7
        with self.assertRaisesRegex(ValueError, "structural evaluation"):
            self.verify()

    def test_unique_candidate_budget_cannot_count_attempts(self):
        self.search["expansion_round_accounting"]["iteration_unit"] = "evaluated_unique_candidates"
        with self.assertRaisesRegex(ValueError, "expected 10"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
