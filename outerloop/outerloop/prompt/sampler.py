

import logging
import json
import random
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from astevolve.domain.design_action import DESIGN_ACTION_VERSION
from outerloop.config import PromptConfig
from outerloop.prompt.templates import TemplateManager
from outerloop.utils.metrics_utils import (
    format_feature_coordinates,
)
from outerloop.utils.metric_semantics import (
    compare_metrics,
    get_metric_spec,
    observe_metric,
    summarize_comparisons,
)
from outerloop.structured_ast_proposal import (
    STRUCTURED_AST_AUDIT_VERSION,
    STRUCTURED_AST_AUDIT_V2_VERSION,
    STRUCTURED_AST_PROPOSAL_MODE,
    STRUCTURED_AST_PROPOSAL_VERSION,
    parent_evolve_hash,
    structured_ast_audit_prompt_contract,
    structured_ast_prompt_context,
    structured_ast_proposal_constraints,
)
from outerloop.structured_design_action_proposal import (
    STRUCTURED_DESIGN_ACTION_CONTROLS,
    STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS,
    STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE,
    STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION,
)

logger = logging.getLogger(__name__)


class PromptSampler:


    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(custom_template_dir=config.template_dir)


        self.system_template_override = None
        self.user_template_override = None


        if not hasattr(logger, "_prompt_sampler_logged"):
            logger.info("Initialized prompt sampler")
            logger._prompt_sampler_logged = True

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:

        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        feature_dimensions: Optional[List[str]] = None,
        current_changes_description: Optional[str] = None,
        optimizer_memory: Optional[Dict[str, Any]] = None,
        island_role: Optional[Dict[str, Any]] = None,
        global_summary: Optional[Dict[str, Any]] = None,
        hierarchical_design: Optional[Dict[str, Any]] = None,
        structured_design_action_context: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:


        proposal_mode = str(
            getattr(self.config, "proposal_mode", "legacy_diff") or "legacy_diff"
        )
        if template_key:

            user_template_key = template_key
        elif self.user_template_override:

            user_template_key = self.user_template_override
        elif proposal_mode == STRUCTURED_AST_PROPOSAL_MODE:
            user_template_key = "structured_ast_user"
        elif proposal_mode == STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE:
            user_template_key = "structured_design_action_user"
        else:

            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"


        user_template = self.template_manager.get_template(user_template_key)


        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
        else:
            system_message = self.config.system_message

            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        if self.config.programs_as_changes_description:
            if self.config.system_message_changes_description:
                system_message_changes_description = self.config.system_message_changes_description.strip()
            else:
                system_message_changes_description = self.template_manager.get_template("system_message_changes_description")

            system_message = self.template_manager.get_template("system_message_with_changes_description").format(
                system_message=system_message,
                system_message_changes_description=system_message_changes_description,
            )

        structured_ast_context = ""
        structured_ast_audit_version = STRUCTURED_AST_AUDIT_VERSION
        structured_ast_audit_contract = structured_ast_audit_prompt_contract(
            structured_ast_audit_version
        )
        structured_design_action_context_section = ""
        migration_instruction = ""
        if proposal_mode == STRUCTURED_AST_PROPOSAL_MODE:
            if bool(getattr(self.config, "hierarchical_audit_v2", False)):
                structured_ast_audit_version = STRUCTURED_AST_AUDIT_V2_VERSION
            structured_ast_audit_contract = structured_ast_audit_prompt_contract(
                structured_ast_audit_version
            )
            proposal_constraints = structured_ast_proposal_constraints(
                program_artifacts
            )
            contract_handoff = (
                optimizer_memory.get("contract_handoff")
                if isinstance(optimizer_memory, dict)
                else None
            )
            structured_ast_context = structured_ast_prompt_context(
                current_program,
                contract_handoff=(
                    contract_handoff
                    if isinstance(contract_handoff, dict)
                    else None
                ),
                proposal_constraints=proposal_constraints,
            )
            scheduler = (
                optimizer_memory.get("scheduler", {})
                if isinstance(optimizer_memory, dict)
                else {}
            )
            score_trend = (
                scheduler.get("score_trend", {})
                if isinstance(scheduler, dict)
                else {}
            )
            if bool(score_trend.get("stagnated")):
                migration_instruction = (
                    "STAGNATION MIGRATION CONTROL (mandatory): choose exactly one "
                    "legal compiled segment marked non-incumbent by MIGRATION_FRONTIER; "
                    "keep at least one incumbent node unchanged as an anchor; activate "
                    "4-8 unique positions total. A structural node must stay wholly "
                    "inside one compiled segment, so create a separate node for the new "
                    "segment. Cite every selected residue evidence_id."
                    " Treat variant_confounded contact differences only as testable "
                    "hypotheses, never as causal attribution."
                )
            else:
                migration_instruction = (
                    "MIGRATION CONTROL: create/delete/migrate/resize/rewire nodes or "
                    "revise amino-acid preferences when evidence supports it. Keep every "
                    "node inside one compiled segment and activate only 4-8 positions. "
                    "When migrating under future stagnation, probe exactly one new "
                    "segment while keeping an incumbent anchor. Treat "
                    "variant_confounded contact differences only as hypotheses."
                )
            system_message = (
                system_message.rstrip()
                + "\n\nStructured AST response mode is active. Return one strict JSON "
                + f"{STRUCTURED_AST_PROPOSAL_VERSION} envelope only. Provide a concise "
                + f"{structured_ast_audit_version} audit record, not hidden chain-of-thought. "
                + "Never emit SEARCH/REPLACE blocks, credentials, paths, provider settings, "
                + "Markdown fences, or prose outside the JSON object."
            )
        elif proposal_mode == STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE:
            design_action_context = (
                deepcopy(dict(structured_design_action_context))
                if isinstance(structured_design_action_context, Mapping)
                else structured_design_action_context
            )
            contract_handoff = (
                optimizer_memory.get("contract_handoff")
                if isinstance(optimizer_memory, dict)
                else None
            )
            if (
                isinstance(design_action_context, dict)
                and isinstance(contract_handoff, Mapping)
            ):


                design_action_context["contract_handoff"] = deepcopy(
                    dict(contract_handoff)
                )
            structured_design_action_context_section = (
                self._format_structured_design_action_context(
                    design_action_context,
                    current_program=current_program,
                )
            )
            system_message = (
                system_message.rstrip()
                + "\n\nStructured DesignAction response mode is active. Return "
                + "exactly one strict JSON "
                + STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION
                + " envelope and nothing else. Never emit Markdown fences, "
                + "SEARCH/REPLACE blocks, Python source, hidden chain-of-thought, "
                + "credentials, paths, provider settings, comments, or prose "
                + "outside that JSON object."
            )

        primary_name, primary_observation = self._primary_objective(program_metrics)
        if primary_name in {"final_energy", "combined_energy"} and primary_observation.comparable:
            system_message = (
                system_message.rstrip()
                + "\n\nPrimary optimization objective: minimize "
                + f"{primary_name}={primary_observation.value:.4f}. "
                + "Use the original lower-is-better energy shown in metrics and artifacts; "
                + "do not optimize or display its internal negated Legacy fitness adapter."
            )


        metrics_str = self._format_metrics(program_metrics)


        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs, feature_dimensions
        )


        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language, feature_dimensions
        )


        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        optimizer_memory_section = self._format_optimizer_memory(optimizer_memory)
        if hierarchical_design is None and isinstance(optimizer_memory, dict):
            candidate_context = optimizer_memory.get("hierarchical_design")
            if isinstance(candidate_context, dict):
                hierarchical_design = candidate_context
        hierarchical_design_section = self._format_hierarchical_design(
            hierarchical_design
        )
        if optimizer_memory_section and "{optimizer_memory}" not in user_template:
            artifacts_section = (
                artifacts_section + "\n\n" + optimizer_memory_section
                if artifacts_section
                else optimizer_memory_section
            )
        if (
            hierarchical_design_section
            and "{hierarchical_design}" not in user_template
        ):
            artifacts_section = (
                artifacts_section + "\n\n" + hierarchical_design_section
                if artifacts_section
                else hierarchical_design_section
            )


        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)


        feature_dimensions = feature_dimensions or []
        fitness_score = self._primary_score_display(program_metrics)
        feature_coords = format_feature_coordinates(program_metrics, feature_dimensions)


        user_message = user_template.format(
            metrics=metrics_str,
            fitness_score=fitness_score,
            feature_coords=feature_coords,
            feature_dimensions=", ".join(feature_dimensions) if feature_dimensions else "None",
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            optimizer_memory=optimizer_memory_section,
            hierarchical_design=hierarchical_design_section,
            structured_ast_context=structured_ast_context,
            migration_instruction=migration_instruction,
            structured_ast_proposal_version=STRUCTURED_AST_PROPOSAL_VERSION,
            structured_ast_audit_version=structured_ast_audit_version,
            structured_ast_audit_contract=structured_ast_audit_contract,
            structured_design_action_context=structured_design_action_context_section,
            structured_design_action_proposal_version=(
                STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION
            ),
            structured_design_action_controls=json.dumps(
                STRUCTURED_DESIGN_ACTION_CONTROLS,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            design_action_version=DESIGN_ACTION_VERSION,
            **kwargs,
        )

        role = island_role if isinstance(island_role, dict) else {}
        if role:
            role_lines = [
                "\n\n## Assigned island role",
                f"Island {role.get('island', '?')}: {role.get('name', role.get('role_id', 'unspecified'))}",
                f"Role focus: {role.get('focus', '')}",
                "Soft objectives: " + ", ".join(str(item) for item in role.get("soft_objectives", [])),
                "Hard feasibility gates remain global; do not trade them away for this role.",
            ]
            if isinstance(global_summary, dict):
                role_lines.extend([
                    "\n## Global island coordinator summary",
                    json.dumps(global_summary, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                    "Use this summary to preserve useful discoveries from other islands and avoid duplicating their failures.",
                ])
            user_message += "\n".join(role_lines)

        if self.config.programs_as_changes_description:
            user_message = self.template_manager.get_template("user_message_with_changes_description").format(
                user_message=user_message,
                changes_description=current_changes_description.rstrip(),
            )

        return {
            "system": system_message,
            "user": user_message,
        }

    def _format_structured_design_action_context(
        self,
        context: Optional[Mapping[str, Any]],
        *,
        current_program: str,
    ) -> str:


        if not isinstance(context, Mapping) or not context:
            raise ValueError(
                "structured_design_action_v1 requires a non-empty "
                "structured_design_action_context mapping"
            )
        payload = deepcopy(dict(context))
        parent = payload.get("parent_binding")
        if not isinstance(parent, Mapping):
            raise ValueError(
                "structured_design_action_context.parent_binding must be a mapping"
            )
        parent_fields = set(parent)
        if parent_fields != STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS:
            unknown = sorted(
                parent_fields - STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS
            )
            missing = sorted(
                STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS - parent_fields
            )
            raise ValueError(
                "structured_design_action_context.parent_binding must contain "
                "exactly the six v1 fields; "
                f"unknown={unknown}; missing={missing}"
            )
        invalid_parent_fields = sorted(
            field
            for field, value in parent.items()
            if not isinstance(value, str) or not value.strip()
        )
        if invalid_parent_fields:
            raise ValueError(
                "structured_design_action_context.parent_binding fields must "
                f"be non-empty strings: {invalid_parent_fields}"
            )
        if not current_program:
            raise ValueError(
                "structured_design_action_v1 requires the exact selected parent program"
            )
        exact_evolve_hash = parent_evolve_hash(current_program)
        if parent.get("parent_evolve_hash") != exact_evolve_hash:
            raise ValueError(
                "structured_design_action_context parent_evolve_hash does not "
                "match the selected parent program"
            )
        supplied_controls = payload.get("required_controls")
        if supplied_controls is not None and supplied_controls != (
            STRUCTURED_DESIGN_ACTION_CONTROLS
        ):
            raise ValueError(
                "structured_design_action_context.required_controls does not "
                "match controller controls"
            )
        payload["parent_evolve_hash"] = exact_evolve_hash
        payload["required_controls"] = deepcopy(
            STRUCTURED_DESIGN_ACTION_CONTROLS
        )
        current_plan = payload.get("current_ast_revision_plan")
        if isinstance(current_plan, Mapping):
            current_edges = current_plan.get("mapping_edges")
            if isinstance(current_edges, list):
                required_functional_ids = sorted(
                    {
                        str(edge.get("functional_node_id"))
                        for edge in current_edges
                        if isinstance(edge, Mapping)
                        and edge.get("functional_node_id")
                    }
                )
                payload["mapping_coverage_contract"] = {
                    "required_functional_node_ids": required_functional_ids,
                    "coverage": "at_least_one_mapping_edge_per_functional_node",
                    "unrelated_current_edges": "copy_byte_for_byte",
                    "edge_removal_requires_explicit_valid_replacement": True,
                    "uncovered_functional_node_policy": "reject_before_gpu",
                }
        payload["lineage_action_union_contract"] = {
            "retain": {
                "replacement_residue": None,
                "target_node_id": "exactly_source_node_id",
            },
            "reassign_node": {
                "replacement_residue": None,
                "target_node_id": "different_live_node_id",
            },
            "replace": {
                "replacement_residue": "different_canonical_amino_acid",
                "target_node_id": "live_owner_node_id",
            },
            "revert_to_wt": {
                "replacement_residue": None,
                "target_node_id": None,
            },
            "action_tokens_are_never_residue_values": True,
        }
        declared_capabilities = payload.get("step1_execution_capabilities")
        declared_capabilities = (
            declared_capabilities
            if isinstance(declared_capabilities, Mapping)
            else {}
        )
        payload["step1_capabilities"] = {
            "lineage_actions": [
                "retain",
                "reassign_node",
                "replace",
                "revert_to_wt",
            ],
            "position_distributions": (
                "executable_exact_prior"
                if bool(declared_capabilities.get("position_distributions"))
                else "must_be_empty"
            ),
            "mutation_modules": (
                "executable_fail_closed"
                if bool(declared_capabilities.get("mutation_modules"))
                else "must_be_empty"
            ),
            "sequence_seeds": (
                "executable_fail_closed"
                if bool(declared_capabilities.get("sequence_seeds"))
                else "must_be_empty"
            ),
            "candidate_portfolio": (
                "executable_fail_closed"
                if bool(declared_capabilities.get("candidate_portfolio"))
                else "must_be_empty"
            ),
        }
        try:
            rendered = json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "structured_design_action_context must be canonical JSON data"
            ) from error
        return (
            "# Controller-Owned Structured DesignAction Context\n"
            "Copy bindings and controls exactly. Do not reinterpret, truncate, "
            "or modify this context.\n"
            + rendered
        )

    def _format_hierarchical_design(
        self, context: Optional[Dict[str, Any]]
    ) -> str:


        if not isinstance(context, dict) or not context:
            return ""
        budget = max(
            4096,
            int(
                getattr(
                    self.config,
                    "hierarchical_design_max_chars",
                    32 * 1024,
                )
                or 0
            ),
        )
        evidence_bundle = context.get("evidence_bundle")
        records = (
            evidence_bundle.get("records", [])
            if isinstance(evidence_bundle, dict)
            else []
        )


        prioritized = {
            "schema_version": context.get("schema_version"),
            "strategy_epoch_id": context.get("strategy_epoch_id"),
            "strategy_hash": context.get("strategy_hash"),
            "strategy_source": context.get("strategy_source"),
            "governance": context.get("governance"),
            "strategy_plan": context.get("strategy_plan"),
            "hypothesis_ledger": context.get("hypothesis_ledger"),
            "reasoning_audit": context.get("reasoning_audit"),
            "evidence_bundle": {
                "schema_version": (
                    evidence_bundle.get("schema_version")
                    if isinstance(evidence_bundle, dict)
                    else None
                ),
                "bundle_hash": (
                    evidence_bundle.get("bundle_hash")
                    if isinstance(evidence_bundle, dict)
                    else None
                ),
                "evidence_index": [
                    {
                        key: item.get(key)
                        for key in (
                            "evidence_id",
                            "kind",
                            "provenance_class",
                            "source_program_id",
                            "payload_hash",
                        )
                    }
                    for item in records
                    if isinstance(item, dict)
                ],
                "records": records,
            },
        }
        return (
            "# Hierarchical Protein Architecture Control\n"
            "This context is controller-owned and frozen for the proposal wave. "
            "Every structural claim must cite one supplied evidence ID. Treat "
            "computed facts as authoritative; do not overwrite measurements, "
            "hashes, protected regions, hard gates, or rollback anchors.\n"
            + self._json_preview(prioritized, max_chars=budget, compact=False)
        )

    def _primary_objective(self, metrics: Dict[str, Any]):
        for name in ("final_energy", "combined_energy", "combined_score"):
            if name not in metrics:
                continue
            observation = observe_metric(name, metrics[name])
            if observation.comparable:
                return name, observation
        return "combined_score", observe_metric("combined_score")

    def _primary_score_display(self, metrics: Dict[str, Any]) -> str:
        name, observation = self._primary_objective(metrics)
        if not observation.comparable:
            return "N/A"
        return f"{observation.value:.4f} ({name}; {observation.spec.direction})"

    def _metric_display(self, name: str, value: Any) -> str:
        observation = observe_metric(name, value)
        direction = observation.spec.direction
        if observation.value is not None:
            return f"{observation.value:.4f} ({direction})"
        return f"N/A ({direction}; {observation.reason or observation.status})"

    def _format_metrics(self, metrics: Dict[str, float]) -> str:

        formatted_parts = []
        if "combined_score" not in metrics:
            formatted_parts.append(
                "- combined_score: N/A (maximize; missing—not comparable)"
            )
        for name, value in metrics.items():
            formatted_parts.append(f"- {name}: {self._metric_display(name, value)}")
        return "\n".join(formatted_parts)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:


        improvement_areas = []
        feature_dimensions = feature_dimensions or []


        if previous_programs:
            prev_metrics = previous_programs[-1].get("metrics", {})
            names = (set(prev_metrics) | set(metrics)) - set(feature_dimensions)
            comparisons = compare_metrics(prev_metrics, metrics, names=names)
            summary = summarize_comparisons(comparisons)
            improved = [item.metric for item in comparisons if item.outcome == "improved"]
            regressed = [item.metric for item in comparisons if item.outcome == "regressed"]
            unchanged = [item.metric for item in comparisons if item.outcome == "unchanged"]
            if improved:
                improvement_areas.append(
                    "Registered directional metrics improved: " + ", ".join(improved) + "."
                )
            if regressed:
                improvement_areas.append(
                    "Registered directional metrics regressed: " + ", ".join(regressed) + "."
                )
            if unchanged and not improved and not regressed:
                improvement_areas.append(
                    "Registered directional metrics were unchanged: " + ", ".join(unchanged) + "."
                )
            if summary["comparable_count"] == 0:
                improvement_areas.append(
                    "No comparable registered directional evidence is available; do not infer improvement from missing or diagnostic values."
                )


        if feature_dimensions:
            feature_coords = format_feature_coordinates(metrics, feature_dimensions)
            if feature_coords == "":
                msg = self.template_manager.get_fragment("no_feature_coordinates")
            else:
                msg = self.template_manager.get_fragment(
                    "exploring_region", features=feature_coords
                )
            improvement_areas.append(msg)


        threshold = (
            self.config.suggest_simplification_after_chars or self.config.code_length_threshold
        )
        if threshold and len(current_program) > threshold:
            msg = self.template_manager.get_fragment("code_too_long", threshold=threshold)
            improvement_areas.append(msg)


        if not improvement_areas:
            improvement_areas.append(self.template_manager.get_fragment("no_specific_guidance"))

        return "\n".join(f"- {area}" for area in improvement_areas)

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:


        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")


        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = (
                program.get("changes_description")
                or program.get("metadata", {}).get(
                    "changes", self.template_manager.get_fragment("attempt_unknown_changes")
                )
            )


            performance_parts = [
                f"{name}: {self._metric_display(name, value)}"
                for name, value in program.get("metrics", {}).items()
            ]
            if "combined_score" not in program.get("metrics", {}):
                performance_parts.insert(
                    0, "combined_score: N/A (maximize; missing—not comparable)"
                )
            performance_str = ", ".join(performance_parts)


            parent_metrics = program.get("metadata", {}).get("parent_metrics", {})
            outcome = self.template_manager.get_fragment("attempt_mixed_metrics")

            program_metrics = program.get("metrics", {})
            names = (set(parent_metrics) | set(program_metrics)) - set(feature_dimensions or [])
            comparison_summary = summarize_comparisons(
                compare_metrics(parent_metrics, program_metrics, names=names)
            )
            if comparison_summary["overall_outcome"] == "all_improved":
                outcome = self.template_manager.get_fragment("attempt_all_metrics_improved")
            elif comparison_summary["overall_outcome"] == "all_regressed":
                outcome = self.template_manager.get_fragment("attempt_all_metrics_regressed")

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )


        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            use_changes = self.config.programs_as_changes_description
            program_code = (
                program.get("changes_description", "")
                if use_changes
                else program.get("code", "")
            )
            if not program_code:
                program_code = "<missing changes_description>" if use_changes else ""

            score = self._primary_score_display(program.get("metrics", {}))


            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    key_features.append(f"{name}: {self._metric_display(name, value)}")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=score,
                    language=("text" if self.config.programs_as_changes_description else language),
                    program_snippet=program_code,
                    key_features=key_features_str,
                )
                + "\n\n"
            )


        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):

            remaining_programs = top_programs[self.config.num_top_programs :]


            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:

                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## " + self.template_manager.get_fragment("diverse_programs_title") + "\n\n"

                for i, program in enumerate(diverse_programs):
                    use_changes = self.config.programs_as_changes_description
                    program_code = (
                        program.get("changes_description", "")
                        if use_changes
                        else program.get("code", "")
                    )
                    if not program_code:
                        program_code = "<missing changes_description>" if use_changes else ""

                    score = self._primary_score_display(program.get("metrics", {}))


                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            self.template_manager.get_fragment("diverse_program_metrics_prefix") + f" {name}"
                            for name in list(program.get("metrics", {}).keys())[
                                :2
                            ]
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=score,
                            language=("text" if self.config.programs_as_changes_description else language),
                            program_snippet=program_code,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )


        combined_programs_str = top_programs_str + diverse_programs_str


        inspirations_section_str = self._format_inspirations_section(
            inspirations, language, feature_dimensions
        )


        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self,
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:

        if not inspirations:
            return ""


        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")

        inspiration_programs_str = ""

        for i, program in enumerate(inspirations):
            use_changes = self.config.programs_as_changes_description
            program_code = (
                program.get("changes_description", "")
                if use_changes
                else program.get("code", "")
            )
            if not program_code:
                program_code = "<missing changes_description>" if use_changes else ""

            score = self._primary_score_display(program.get("metrics", {}))


            program_type = self._determine_program_type(program, feature_dimensions or [])


            unique_features = self._extract_unique_features(program)

            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=score,
                    program_type=program_type,
                    language=("text" if self.config.programs_as_changes_description else language),
                    program_snippet=program_code,
                    unique_features=unique_features,
                )
                + "\n\n"
            )

        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )

    def _determine_program_type(
        self, program: Dict[str, Any], feature_dimensions: Optional[List[str]] = None
    ) -> str:

        metadata = program.get("metadata", {})

        if metadata.get("diverse", False):
            return self.template_manager.get_fragment("inspiration_type_diverse")
        if metadata.get("migrant", False):
            return self.template_manager.get_fragment("inspiration_type_migrant")
        if metadata.get("random", False):
            return self.template_manager.get_fragment("inspiration_type_random")


        return self.template_manager.get_fragment("inspiration_type_score_exploratory")

    def _extract_unique_features(self, program: Dict[str, Any]) -> str:

        features = []


        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if (
                isinstance(changes, str)
                and self.config.include_changes_under_chars
                and len(changes) < self.config.include_changes_under_chars
            ):
                features.append(self.template_manager.get_fragment("inspiration_changes_prefix").format(changes=changes))


        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            observation = observe_metric(metric_name, value)
            if observation.value is not None:
                features.append(
                    f"{metric_name}={observation.value:.3f} ({observation.spec.direction})"
                )


        code = program.get("code", "")
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append(self.template_manager.get_fragment("inspiration_code_with_class"))
            if "numpy" in code_lower or "np." in code_lower:
                features.append(self.template_manager.get_fragment("inspiration_code_with_numpy"))
            if "for" in code_lower and "while" in code_lower:
                features.append(self.template_manager.get_fragment("inspiration_code_with_mixed_iteration"))
            if (
                self.config.concise_implementation_max_lines
                and len(code.split("\n")) <= self.config.concise_implementation_max_lines
            ):
                features.append(self.template_manager.get_fragment("inspiration_code_with_concise_line"))
            elif (
                self.config.comprehensive_implementation_min_lines
                and len(code.split("\n")) >= self.config.comprehensive_implementation_min_lines
            ):
                features.append(self.template_manager.get_fragment("inspiration_code_with_comprehensive_line"))


        if not features:
            program_type = self._determine_program_type(program)
            features.append(self.template_manager.get_fragment("inspiration_no_features_postfix").format(program_type=program_type))


        feature_limit = self.config.num_top_programs
        return ", ".join(features[:feature_limit])

    def _apply_template_variations(self, template: str) -> str:

        result = template


        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:

        if not artifacts:
            return ""

        title = "## " + self.template_manager.get_fragment("artifact_title")
        per_artifact_limit = max(0, int(self.config.max_artifact_bytes or 0))
        total_limit_raw = getattr(self.config, "max_artifact_total_bytes", None)
        complete = {
            str(key)
            for key in (getattr(self.config, "complete_artifacts", None) or [])
            if str(key) in artifacts
        }

        def byte_count(value: str) -> int:
            return len(value.encode("utf-8"))

        def byte_prefix(value: str, limit: int) -> str:
            if limit <= 0:
                return ""
            return value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")

        marker = "\n... (truncated)"

        def bounded_content(value: str, limit: int) -> str:
            if byte_count(value) <= limit:
                return value
            marker_size = byte_count(marker)
            if limit > marker_size:
                return byte_prefix(value, limit - marker_size) + marker
            return byte_prefix(value, limit)


        if total_limit_raw is None:
            sections = []
            for key, value in artifacts.items():
                content = self._safe_decode_artifact(value)
                if (
                    str(key) in complete
                    and per_artifact_limit
                    and byte_count(content) > per_artifact_limit
                ):
                    raise ValueError(
                        f"complete artifact {key!r} exceeds max_artifact_bytes"
                    )
                if (
                    str(key) not in complete
                    and per_artifact_limit
                    and len(content) > per_artifact_limit
                ):
                    content = content[:per_artifact_limit] + "\n... (truncated)"
                sections.append(f"### {key}\n```\n{content}\n```")
            return title + "\n\n" + "\n\n".join(sections) if sections else ""

        total_limit = max(0, int(total_limit_raw or 0))

        priority = [
            str(key)
            for key in (getattr(self.config, "artifact_priority", None) or [])
            if str(key) in artifacts
        ]
        ordered_keys = list(dict.fromkeys(priority))
        ordered_keys.extend(key for key in artifacts if key not in ordered_keys)
        contents = {
            key: self._safe_decode_artifact(artifacts[key]) for key in ordered_keys
        }

        def section_parts(key: str) -> tuple[str, str, str]:
            return "\n\n", f"### {key}\n```\n", "\n```"

        complete_section_sizes = {}
        for key in ordered_keys:
            if str(key) not in complete:
                continue
            content_size = byte_count(contents[key])
            if per_artifact_limit and content_size > per_artifact_limit:
                raise ValueError(
                    f"complete artifact {key!r} exceeds max_artifact_bytes"
                )
            complete_section_sizes[key] = sum(
                byte_count(part) for part in (*section_parts(key), contents[key])
            )
        required_complete_bytes = byte_count(title) + sum(
            complete_section_sizes.values()
        )
        if required_complete_bytes > total_limit:
            raise ValueError(
                "complete artifact budget cannot preserve all configured evidence: "
                f"requires {required_complete_bytes} bytes, received {total_limit}"
            )
        if total_limit <= byte_count(title):
            return byte_prefix(title, total_limit)

        rendered = title
        for index, key in enumerate(ordered_keys):
            separator, section_prefix, section_suffix = section_parts(key)
            overhead = sum(
                byte_count(part) for part in (separator, section_prefix, section_suffix)
            )
            reserved_after = sum(
                complete_section_sizes.get(later_key, 0)
                for later_key in ordered_keys[index + 1 :]
            )
            remaining = (
                total_limit - byte_count(rendered) - overhead - reserved_after
            )
            if remaining <= 0:
                if str(key) in complete:
                    raise ValueError(
                        f"complete artifact {key!r} cannot fit aggregate budget"
                    )
                continue

            content = contents[key]
            content_limit = remaining
            if per_artifact_limit:
                content_limit = min(content_limit, per_artifact_limit)
            if str(key) in complete:
                if byte_count(content) > content_limit:
                    raise ValueError(
                        f"complete artifact {key!r} cannot fit aggregate budget"
                    )
            else:
                content = bounded_content(content, content_limit)

            rendered += separator + section_prefix + content + section_suffix

        if byte_count(rendered) > total_limit:
            raise ValueError("rendered artifact section exceeds aggregate byte budget")
        return rendered

    def _format_optimizer_memory(self, memory: Optional[Dict[str, Any]]) -> str:

        if not memory:
            return ""

        mandatory_sections = []
        optional_sections = []
        case_memory = memory.get("case_memory", {}) if isinstance(memory, dict) else {}
        recent = (
            memory.get("recent_attempts")
            or memory.get("trajectory_memory", [])
            if isinstance(memory, dict)
            else []
        )
        top = (
            memory.get("top_programs")
            or memory.get("best_candidate_memory", [])
            if isinstance(memory, dict)
            else []
        )
        scheduler = memory.get("scheduler", {}) if isinstance(memory, dict) else {}
        contract_handoff = memory.get("contract_handoff", {}) if isinstance(memory, dict) else {}
        capsule = (
            memory.get("mandatory_prompt_capsule", {})
            if isinstance(memory, dict)
            else {}
        )

        if contract_handoff and (
            str(getattr(self.config, "proposal_mode", "legacy_diff"))
            not in {
                STRUCTURED_AST_PROPOSAL_MODE,
                STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE,
            }
        ):
            mandatory_sections.append(
                "## Required Parent EditContract Response\n"
                "This is a bounded, hash-bound projection of the exact parent "
                "contract; runtime validation retains the full canonical envelope. "
                "Set last_contract_response to the supplied closed response schema; "
                "do not copy or rewrite edit_contract itself.\n"
                + self._json_preview(contract_handoff, max_chars=0, compact=True)
            )
        if capsule:
            mandatory_sections.append(
                "## Mandatory Recent Evidence\n"
                "This capsule is chronological evidence, not a best-score ranking.\n"
                + self._json_preview(capsule, max_chars=0, compact=True)
            )
        if scheduler:
            optional_sections.append(("Scheduler Control", scheduler))
        if case_memory:
            optional_sections.append(("Case Memory", case_memory))
        if recent:
            optional_sections.append(("Chronological Recent Attempts", recent[-8:]))
        if top:
            optional_sections.append(("Top Programs (separate ranking)", top[:5]))

        if not mandatory_sections and not optional_sections:
            return ""
        prefix = "# LLM Optimizer Memory\n"
        mandatory = "\n\n".join(mandatory_sections)
        rendered = prefix + mandatory if mandatory else prefix.rstrip()
        max_chars = int(
            getattr(self.config, "optimizer_memory_max_chars", 12 * 1024) or 0
        )


        remaining = 0 if max_chars <= 0 else max(0, max_chars - len(rendered) - 2)
        for title, payload in optional_sections:
            header = f"## {title}\n"
            if max_chars > 0 and remaining <= len(header) + 24:
                break
            preview_budget = 0 if max_chars <= 0 else remaining - len(header)
            section = header + self._json_preview(
                payload,
                max_chars=preview_budget,
                compact=False,
            )
            candidate = rendered + "\n\n" + section
            if max_chars > 0 and len(candidate) > max_chars:


                available = max_chars - len(rendered) - 2
                if available <= len(header):
                    break
                section = section[:available]
                candidate = rendered + "\n\n" + section
            rendered = candidate
            if max_chars > 0:
                remaining = max(0, max_chars - len(rendered) - 2)
        return rendered

    def _json_preview(
        self,
        value: Any,
        *,
        max_chars: Optional[int] = None,
        compact: bool = False,
    ) -> str:

        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
                default=str,
            )
        except TypeError:
            text = str(value)
        if max_chars is None:
            max_chars = int(
                getattr(self.config, "optimizer_memory_max_chars", 12 * 1024) or 0
            )
        if max_chars > 0 and len(text) > max_chars:
            marker = "\n... (optimizer memory truncated)"
            keep = max(0, int(max_chars) - len(marker))
            text = text[:keep] + marker
        return "```json\n" + text + "\n```"

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:

        if isinstance(value, str):

            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        elif isinstance(value, (dict, list, tuple)):
            try:
                decoded = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=str,
                )
            except (TypeError, ValueError):
                decoded = str(value)
            if self.config.artifact_security_filter:
                return self._apply_security_filter(decoded)
            return decoded
        else:
            return str(value)

    def _apply_security_filter(self, text: str) -> str:

        import re


        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)


        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
