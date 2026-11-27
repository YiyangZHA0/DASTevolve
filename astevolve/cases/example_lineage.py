"""A short, measured parent-to-child run for the packaged examples."""

import json
import logging
from pathlib import Path

from engine.experiment_identity import SequenceBundleIdentity
from outerloop.controller import OuterLoop


def verified_stage(program, artifacts, parent, expected_rounds):
    """Verify the selected trial's actual search and its inherited root."""
    accepted = program.metadata["accepted_runtime_artifact"]
    if accepted["acceptance"]["status"] != "accepted":
        raise ValueError(f"Program {program.id} has no accepted trial")
    search = artifacts["search_artifacts"]
    if isinstance(search, (str, bytes)):
        search = json.loads(search)
    root = SequenceBundleIdentity.create(search["root_candidate"]["seqs"])
    winner = SequenceBundleIdentity.from_mapping(accepted["sequence_bundle_identity"])
    if parent and (program.parent_id != parent["program_id"]
                   or root.sequence_bundle_hash != parent["winner_sequence_hash"]):
        raise ValueError(f"Program {program.id} did not inherit its predecessor's winning sequence")

    accounting = search["expansion_round_accounting"]
    budget = search["mcts_candidate_budget"] or {}
    unique = int(budget["completed_evaluated_unique_candidates"])
    unit = accounting["iteration_unit"]
    if unit == "evaluated_unique_candidates":
        rounds = unique
    elif unit == "expansion_rounds":
        rounds = accounting["physical_rounds"]
    else:
        raise ValueError(f"Unknown inner search budget unit: {unit}")
    if rounds != expected_rounds or accounting["requested_rounds"] != expected_rounds:
        raise ValueError(f"Program {program.id} completed {rounds} {unit}, expected {expected_rounds}")
    structures = search["inner_structure_summary"]
    if unique < 1 or structures["evaluated_count"] <= structures["failure_count"]:
        raise ValueError(f"Program {program.id} has no successful structural evaluation of a candidate")
    quality = search["mcts_tree_quality"]
    if quality["required"] and not quality["pass"]:
        raise ValueError(f"Program {program.id} failed the case's tree quality requirements")
    return {
        "program_id": program.id, "parent_id": program.parent_id,
        "proposal_iteration": program.iteration_found,
        "root_sequence_hash": root.sequence_bundle_hash,
        "winner_sequence_hash": winner.sequence_bundle_hash,
        "inner_unit": unit, "completed": rounds, "scored_candidates": unique,
        "hard_gate_pass": program.metrics.get("hard_gate_pass"),
        "final_energy": program.metrics.get("final_energy"),
    }


class ExampleLineageRun(OuterLoop):
    """Use a single chronological lineage in the short example profile."""

    def _select_serial_parent(self, island_id):
        return self.lineage_parent, []

    async def _run_evolution_serial_with_checkpoints(self, start_iteration, max_iterations, target_score):
        (self.lineage_parent,) = self.database.programs.values()
        stages = [verified_stage(self.lineage_parent, self.database.get_artifacts(self.lineage_parent.id),
                                 None, self.inner_rounds)]
        report = Path(self.output_dir) / "verified-rounds.json"

        def record():
            temporary = report.with_suffix(".tmp")
            temporary.write_text(json.dumps({"requested_revisions": max_iterations, "stages": stages},
                                             indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(report)

        record()
        for iteration in range(start_iteration, start_iteration + self.proposal_limit):
            if self._shutdown_requested:
                break
            await super()._run_evolution_serial_with_checkpoints(iteration, 1, None)
            children = [p for p in self.database.programs.values()
                        if p.iteration_found == iteration and p.parent_id == self.lineage_parent.id]
            if not children:
                logging.getLogger(__name__).warning("No accepted child at proposal %s", iteration)
                continue
            (child,) = children
            stages.append(verified_stage(child, self.database.get_artifacts(child.id),
                                         stages[-1], self.inner_rounds))
            self.lineage_parent = child
            record()
            if len(stages) == max_iterations + 1:
                return
        raise RuntimeError(f"Completed {len(stages) - 1}/{max_iterations} verified outer revisions; see {report}")
