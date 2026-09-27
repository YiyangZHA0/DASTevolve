"""Offline regression checks for the actual MCTS selection/expansion loop."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from astevolve.core.protein_lang import Segment
from astevolve.search import inner_opt
from astevolve.search.config import SAConfig
from astevolve.search import mcts_candidate_expansion as expansion
from astevolve.search.mcts_fidelity_upgrade import apply_reward_delta, refresh_best_reward
from astevolve.search.reporting import _mcts_select_leaf


class MCTSChecks(unittest.TestCase):
    def run_search(self, *, optimizer=False, sweep=False, evaluator=None, seed=77, rounds=40):
        segment = Segment(kind="cdr", name="loop", chain_id="B", spans=[(0, 20)])
        compiled = {"chain_order": ["B"], "chain_lengths": {"B": 20}, "segments": [segment]}
        cfg = SAConfig(
            iterations=rounds, search_method="mcts", mcts_max_depth=4,
            progen_weight=0.0, fast_filter_enabled=False,
            inner_structure_enabled=True,
            mcts_save_tree=False, mcts_save_variants=False,
            node_optimizer_enabled=optimizer,
            mcts_progressive_widening_c=0.5,
            mcts_node_sweep_enabled=sweep, mcts_node_sweep_count=rounds if sweep else 0,
            mcts_iteration_unit="evaluated_unique_candidates" if sweep else "expansion_rounds",
        )
        captured = {}
        def artifacts(config, tree, candidates, summary):
            captured["tree"] = deepcopy(tree)
            return {}
        def fast(seqs, *args):
            loss = float(sum(aa != "A" for aa in seqs["B"]))
            return {"total": loss}, {"loglik_sum": 0.0, "loglik_avg": 0.0}, loss, False
        def structure(candidate):
            count = sum(aa != "A" for aa in candidate["seqs"]["B"])
            loss, passes = evaluator(candidate, count) if evaluator else (-float(count), True)
            return {
                "status": "ok", "gate_pass": passes, "selection_loss": loss,
                "structure_combined_energy": loss,
                "result": {"inner_evaluator_report": {"hard_gate_pass": passes}},
            }
        serial = 0
        def proposals(seqs, segment, positions, rng, cfg, masks, memory, **kwargs):
            nonlocal serial
            serial += 1
            values = list(seqs["B"])
            # A finite deterministic, legal proposal stream; scientific models
            # are replaced, while selection, expansion, gates and backup are real.
            pos = (serial - 1) % 20
            old = values[pos]
            values[pos] = "CDEFGHIKLMNPQRSTVWY"[(serial // 20) % 19]
            if values[pos] == old:
                values[pos] = "Y" if old != "Y" else "V"
            return [({"B": "".join(values)}, {
                "op": "point", "node": "loop", "chain_id": "B",
                "target_nodes": ["loop"], "positions": {"B": [pos]},
                "changes": [{"chain_id": "B", "position": pos, "before": old, "after": values[pos]}],
                "mutation_plan": {"tier": "explore"},
            })]
        actions = [{"action_id": "edit", "compiled_segment_name": "loop"}] if sweep else []
        with patch.object(inner_opt, "_score_fast_with_run_memory", side_effect=fast), \
             patch.object(inner_opt, "_write_inner_loop_artifacts", side_effect=artifacts), \
             patch.object(inner_opt, "_active_mapping_actions", return_value=actions), \
             patch.object(expansion, "mutate_node_candidates", side_effect=proposals):
            out = inner_opt._run_mcts_search(
                compiled, [], cfg, {"B": np.ones(20, dtype=bool)},
                np.random.default_rng(seed), {"B": "A" * 20}, {"B": {}}, None,
                inner_structure_evaluator=structure,
            )
        return out, captured["tree"]

    def test_real_search_reaches_depth_and_starts_a_fresh_tree(self):
        for optimizer in (False, True):
            first, tree = self.run_search(optimizer=optimizer)
            second, other = self.run_search(optimizer=optimizer)
            self.assertGreaterEqual(max(n["depth"] for n in tree.values()), 3)
            self.assertLessEqual(max(n["depth"] for n in tree.values()), 4)
            self.assertGreater(len(tree["root"]["children"]), 1)
            self.assertEqual(tree, other)
            self.assertEqual(tree["root"]["visits"], len(tree) - 1)
            self.assertEqual(first[6]["best_path"], second[6]["best_path"])

    def test_feedback_changes_branch_selection(self):
        good, good_tree = self.run_search(optimizer=True, evaluator=lambda c, n: (-float(n), True))
        bad, bad_tree = self.run_search(optimizer=True, evaluator=lambda c, n: (float(n), True))
        self.assertNotEqual(
            [c["parent_id"] for c in good[5]], [c["parent_id"] for c in bad[5]]
        )
        self.assertNotEqual(good_tree, bad_tree)

    def test_node_sweep_respects_depth_limit(self):
        out, tree = self.run_search(sweep=True)
        self.assertLessEqual(max(n["depth"] for n in tree.values()), 4)
        self.assertEqual(out[4]["node_sweep_summary"]["completed_candidates"], 40)
        self.assertTrue(out[4]["depth_limit_parent_redirects"])
        self.assertFalse(out[4]["mcts_candidate_budget"]["root_baseline_counted_in_candidate_budget"])
        self.assertEqual(out[4]["mcts_candidate_budget"]["root_baseline_inline_evaluator_invocations"], 1)

    def test_infeasible_root_cannot_displace_feasible_child(self):
        out, tree = self.run_search(
            evaluator=lambda c, n: (0.0, False) if n == 0 else (10.0 + n, True)
        )
        self.assertNotEqual(out[6]["best_node_id"], "root")
        self.assertTrue(tree[out[6]["best_node_id"]]["inner_structure_gate_pass"])

    def test_failed_gate_never_receives_positive_reward(self):
        out, tree = self.run_search(evaluator=lambda c, n: (-float(n), n == 0))
        failed = [n for name, n in tree.items() if name != "root"]
        self.assertTrue(failed)
        self.assertTrue(all(n["reward"] <= 0 for n in failed))

    def test_high_fidelity_delta_changes_puct_without_extra_visits(self):
        tree = {
            "root": {"id":"root", "parent":None, "children":["a","b"], "depth":0, "visits":10, "total_reward":0.5},
            "a": {"id":"a", "parent":"root", "children":[], "depth":1, "visits":5, "total_reward":0.4, "prior":0.5, "reward":0.8},
            "b": {"id":"b", "parent":"root", "children":[], "depth":1, "visits":5, "total_reward":0.1, "prior":0.5, "reward":-0.2},
        }
        cfg = SAConfig(mcts_max_depth=4, mcts_c_puct=0.0)
        self.assertEqual(_mcts_select_leaf(tree,"root",cfg), "a")
        visits = {k:v["visits"] for k,v in tree.items()}
        # The one evaluated value is corrected once along its ancestor chain.
        apply_reward_delta(tree,"a",old_reward=0.8,new_reward=-0.8)
        tree["a"]["reward"] = -0.8
        refresh_best_reward(tree)
        self.assertEqual(visits,{k:v["visits"] for k,v in tree.items()})
        self.assertEqual(_mcts_select_leaf(tree,"root",cfg), "b")
        self.assertAlmostEqual(tree["root"]["total_reward"], -1.1)

    def test_terminal_best_branch_does_not_hide_expandable_sibling(self):
        tree = {
            "root": {"parent":None,"children":["a","b"],"depth":0,"visits":3,"total_reward":2.0},
            "a": {"parent":"root","children":["terminal"],"depth":1,"visits":2,"total_reward":2.0},
            "terminal": {"parent":"a","children":[],"depth":2,"visits":1,"total_reward":1.0},
            "b": {"parent":"root","children":[],"depth":1,"visits":1,"total_reward":0.0},
        }
        self.assertEqual(_mcts_select_leaf(tree,"root",SAConfig(mcts_max_depth=2)), "b")

    def test_real_fidelity_checkpoint_rejects_proxy_winner(self):
        root = {"variant_id":"root", "seqs":{"B":"AA"}, "fast_loss":0.0, "selection_loss":0.0}
        candidates = [
            {"variant_id": name, "seqs":{"B": sequence}, "fast_loss":loss,
             "selection_loss":loss, "reward":reward, "inner_structure_gate_pass":True,
             "inner_structure_evaluation":{"status":"ok","result":{"inner_evaluator_report":{}}}}
            for name, sequence, loss, reward in [("a","CA",-1.0,0.6),("b","DA",-0.5,0.3)]
        ]
        tree = {"root":{"id":"root","parent":None,"children":["a","b"],"depth":0,"visits":2,"total_reward":0.9}}
        for candidate in candidates:
            tree[candidate["variant_id"]] = {
                **deepcopy(candidate), "parent":"root", "children":[], "depth":1,
                "visits":1,"total_reward":candidate["reward"],"prior":0.5,
            }
        state = expansion.MCTSExpansionState(
            candidate_serial=2, best_sequences={"B":"CA"}, best_breakdown={"total":0},
            best_progen={"loglik_sum":0,"loglik_avg":0}, best_fast=-1.0,
            best_node_id="a",best_selection_loss=-1.0,
        )
        cfg = SAConfig(inner_structure_model="offline", mcts_fidelity_upgrade_candidates=1)
        visits = {key: node["visits"] for key,node in tree.items()}
        seen = []
        def high_fidelity(candidate):
            seen.append(candidate["variant_id"])
            return {"status":"ok", "gate_pass":candidate["variant_id"] != "a",
                    "selection_loss":2.0 if candidate["variant_id"] == "a" else 0.0}
        history = {}
        inner_opt._run_mcts_fidelity_checkpoint(
            tree=tree,candidates=candidates,root_candidate=root,state=state,
            evaluator=high_fidelity,cfg=cfg,reward_scale=1.0,history=history,
            checkpoint=2,wave_start=0,wave_end=2,
        )
        self.assertEqual(seen,["root","a"])
        self.assertEqual(state.best_node_id,"b")
        self.assertEqual(tree["a"]["reward"],-1.0)
        self.assertEqual(visits,{key:node["visits"] for key,node in tree.items()})
        self.assertEqual(_mcts_select_leaf(tree,"root",cfg),"b")

    def test_bootstrap_scope_and_old_signature_compatibility(self):
        masks = {"B": [True, False]}
        fixed = {"B": {1: "A"}}
        score = {"mutation_scope_contract": {"active_positions_by_node": {"new_node": [0]}}}
        base = dict(root_sequences={"B":"AA"}, config={}, count=2, seed=3,
                    masks=masks, fixed_residues=fixed, score_config=score)
        def old(*, root_sequences, config, count, seed):
            return count, seed
        self.assertEqual(inner_opt._call_sequence_bootstrap(old, **base), (2,3))
        def current(*, root_sequences, config, count, seed, masks, fixed_residues, score_config):
            self.assertEqual(masks, {"B":[True,False]})
            self.assertEqual(fixed_residues, fixed)
            self.assertEqual(score_config, score)
            masks["B"][0] = False
            score_config["mutation_scope_contract"].clear()
            return "valid"
        self.assertEqual(inner_opt._call_sequence_bootstrap(current, **base), "valid")
        self.assertEqual(masks, {"B":[True,False]})
        self.assertTrue(score["mutation_scope_contract"])
        calls = []
        def broken(**kwargs):
            calls.append(kwargs)
            raise TypeError("provider-internal failure")
        with self.assertRaisesRegex(TypeError, "provider-internal failure"):
            inner_opt._call_sequence_bootstrap(broken, **base)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
