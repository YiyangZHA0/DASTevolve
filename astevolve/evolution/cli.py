

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

from astevolve.cases.registry import resolve_case

from .archive import ArchiveProjection, ArchiveProjectionConfig, FeatureDimension
from .memory import EvolutionMemoryProjection, MemoryProjectionConfig
from .evaluation_cache import (
    CachedProposalEvaluator,
    ContentDeterminismDeclaration,
    EvaluationIdentityPolicy,
    FailureCachePolicy,
    FileEvaluationCacheStore,
)
from .orchestrator import (
    EvolutionInputSnapshot,
    EvolutionOrchestrationError,
    EvolutionOrchestrator,
    EvolutionRunConfig,
)
from .execution_identity import (
    ExecutionIdentityError,
    NATIVE_EXECUTION_CONTRACT_KEY,
    NATIVE_EXECUTION_CONTRACT_VERSION,
    absent_resource_identity,
    canonical_payload_sha256,
    factory_code_identity,
    python_source_tree_identity,
    resource_content_identity,
    seal_execution_contract,
)
from .persistence import FileGenerationLedger
from .projections import CompositeCommitProjection
from .workers import InlineEvaluationExecutor, OwnedProcessEvaluationExecutor


NATIVE_RUN_REPORT_VERSION = "astevolve.native_evolution_run_report.v1"
_EXECUTION_CONTRACT_KEY = NATIVE_EXECUTION_CONTRACT_KEY

_BUILTIN_PROPOSAL_SOURCES = {
    "configured_llm": (
        "astevolve.adapters.evolution:create_configured_llm_proposal_source"
    ),
}
_BUILTIN_EVALUATORS = {
    "case_design": "astevolve.adapters.evolution:create_case_design_evaluator",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_PYTHON_SOURCE_ROOTS = {
    "astevolve": _PROJECT_ROOT / "astevolve",
    "engine": _PROJECT_ROOT / "engine",
    "outerloop": _PROJECT_ROOT / "outerloop",
}
_CASE_DESIGN_FACTORY_TARGET = (
    "astevolve.adapters.evolution.case_runtime",
    "create_case_design_evaluator",
)
_PROJECT_BUILTIN_FACTORY_TARGETS = frozenset(
    {
        (
            "astevolve.adapters.evolution.configured_llm",
            "create_configured_llm_proposal_source",
        ),
        _CASE_DESIGN_FACTORY_TARGET,
    }
)


class NativeCLIError(ValueError):
    pass


@dataclass(frozen=True)
class NativeRuntimeServices:


    input_snapshot: EvolutionInputSnapshot
    run_config: EvolutionRunConfig
    archive: ArchiveProjection
    memory: EvolutionMemoryProjection


def _read_mapping(path: Optional[str], *, label: str) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path).expanduser().resolve()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeCLIError(f"cannot read {label}: {target}") from exc
    if not isinstance(value, dict):
        raise NativeCLIError(f"{label} must contain a JSON object")
    return value


def _load_factory(
    spec: str,
    *,
    role: str,
    payload: Mapping[str, Any],
    services: NativeRuntimeServices,
) -> Any:
    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise NativeCLIError(f"{role} factory must use module:attribute syntax")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise NativeCLIError(f"cannot import {role} factory {spec!r}") from exc
    if not callable(factory):
        raise NativeCLIError(f"{role} factory {spec!r} is not callable")
    try:
        encoded_payload = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        detached_payload = json.loads(encoded_payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NativeCLIError(
            f"{role} payload must contain finite JSON-compatible data"
        ) from exc
    if not isinstance(detached_payload, dict):
        raise NativeCLIError(f"{role} payload must be a JSON object")
    try:
        signature = inspect.signature(factory)
        try:
            signature.bind(detached_payload, services)
            use_services = True
        except TypeError:
            signature.bind(detached_payload)
            use_services = False
        value = (
            factory(detached_payload, services)
            if use_services
            else factory(detached_payload)
        )
    except Exception as exc:
        raise NativeCLIError(f"{role} factory {spec!r} failed") from exc
    method = "propose" if role == "proposal source" else "evaluate"
    if not callable(getattr(value, method, None)):
        raise NativeCLIError(f"{role} factory must return an object with {method}()")
    return value


def _archive_config(value: Mapping[str, Any]) -> ArchiveProjectionConfig:
    allowed = {
        "feature_dimensions",
        "objective_path",
        "objective_direction",
        "num_islands",
        "migration_interval",
        "migrants_per_island",
    }
    unknown = set(value) - allowed
    if unknown:
        raise NativeCLIError(f"unknown archive config fields: {sorted(unknown)}")
    raw_dimensions = value.get("feature_dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise NativeCLIError("archive feature_dimensions must be a non-empty list")
    dimensions = []
    fields = {"name", "report_path", "minimum", "maximum", "bins"}
    for index, raw in enumerate(raw_dimensions):
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise NativeCLIError(
                f"archive feature_dimensions[{index}] has invalid fields"
            )
        path = raw["report_path"]
        if not isinstance(path, list):
            raise NativeCLIError("archive report_path must be a string list")
        dimensions.append(
            FeatureDimension(
                name=raw["name"],
                report_path=tuple(path),
                minimum=raw["minimum"],
                maximum=raw["maximum"],
                bins=raw["bins"],
            )
        )
    objective_path = value.get("objective_path", ["normalized_score"])
    if not isinstance(objective_path, list):
        raise NativeCLIError("archive objective_path must be a string list")
    return ArchiveProjectionConfig(
        feature_dimensions=tuple(dimensions),
        objective_path=tuple(objective_path),
        objective_direction=value.get("objective_direction", "maximize"),
        num_islands=value.get("num_islands", 1),
        migration_interval=value.get("migration_interval", 10),
        migrants_per_island=value.get("migrants_per_island", 1),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the project-owned transactional evolution backend."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--proposal-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-json")
    parser.add_argument("--archive-config")
    parser.add_argument("--report")
    parser.add_argument("--recent-memory", type=int, default=20)
    parser.add_argument("--best-memory", type=int, default=5)
    parser.add_argument("--executor", choices=("inline", "process"), default="inline")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument(
        "--proposal-source",
        choices=tuple(sorted(_BUILTIN_PROPOSAL_SOURCES)),
    )
    parser.add_argument(
        "--proposal-evaluator",
        choices=tuple(sorted(_BUILTIN_EVALUATORS)),
    )
    parser.add_argument("--proposal-source-factory")
    parser.add_argument("--evaluator-factory")
    parser.add_argument(
        "--identity-resource",
        action="append",
        default=[],
        help=(
            "File or directory whose content identity is part of the run; "
            "repeat for plugin-owned external resources."
        ),
    )
    parser.add_argument("--evaluation-cache")
    parser.add_argument("--evaluator-namespace")
    parser.add_argument(
        "--cache-identity",
        choices=("exact_task", "content_deterministic"),
        default="exact_task",
    )
    parser.add_argument("--cache-failures", action="store_true")
    parser.add_argument("--content-determinism-rationale")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_components(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    services: NativeRuntimeServices,
):
    proposal_spec, evaluator_spec = _component_specs(args)
    assert proposal_spec is not None and evaluator_spec is not None
    return (
        _load_factory(
            proposal_spec,
            role="proposal source",
            payload=payload,
            services=services,
        ),
        _load_factory(
            evaluator_spec,
            role="evaluator",
            payload=payload,
            services=services,
        ),
    )


def _component_specs(
    args: argparse.Namespace,
) -> tuple[Optional[str], Optional[str]]:


    if args.proposal_source and args.proposal_source_factory:
        raise NativeCLIError(
            "choose either --proposal-source or --proposal-source-factory"
        )
    if args.proposal_evaluator and args.evaluator_factory:
        raise NativeCLIError(
            "choose either --proposal-evaluator or --evaluator-factory"
        )
    proposal_spec = args.proposal_source_factory or _BUILTIN_PROPOSAL_SOURCES.get(
        args.proposal_source
    )
    evaluator_spec = args.evaluator_factory or _BUILTIN_EVALUATORS.get(
        args.proposal_evaluator
    )
    if not proposal_spec or not evaluator_spec:
        raise NativeCLIError(
            "native runs require one proposal source and evaluator"
        )
    return proposal_spec, evaluator_spec


def _cache_config(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    identity = EvaluationIdentityPolicy(args.cache_identity)
    failure = (
        FailureCachePolicy.CACHE if args.cache_failures else FailureCachePolicy.RETRY
    )
    if not args.evaluation_cache:
        if args.evaluator_namespace or args.content_determinism_rationale:
            raise NativeCLIError(
                "cache namespace/rationale requires --evaluation-cache"
            )
        if identity is not EvaluationIdentityPolicy.EXACT_TASK or args.cache_failures:
            raise NativeCLIError("cache policy flags require --evaluation-cache")
        return identity, failure, None
    if not args.evaluator_namespace:
        raise NativeCLIError("--evaluation-cache requires --evaluator-namespace")
    if (
        not isinstance(args.evaluator_namespace, str)
        or not args.evaluator_namespace.strip()
        or args.evaluator_namespace != args.evaluator_namespace.strip()
    ):
        raise NativeCLIError("--evaluator-namespace must be non-empty and trimmed")
    declaration = None
    if identity is EvaluationIdentityPolicy.CONTENT_DETERMINISTIC:
        if not args.content_determinism_rationale:
            raise NativeCLIError(
                "content_deterministic cache requires "
                "--content-determinism-rationale"
            )
        declaration = ContentDeterminismDeclaration.create(
            evaluator_namespace=args.evaluator_namespace,
            rationale=args.content_determinism_rationale,
        )
    elif args.content_determinism_rationale:
        raise NativeCLIError(
            "--content-determinism-rationale requires content_deterministic identity"
        )
    return identity, failure, declaration


def _identity_resource(
    path: Path,
    *,
    role: str,
    required: bool,
) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise NativeCLIError(f"required identity resource is missing: {role}")
        return absent_resource_identity(role=role)
    try:
        return resource_content_identity(path, role=role)
    except ExecutionIdentityError as exc:
        raise NativeCLIError(f"cannot fingerprint identity resource {role!r}") from exc


def _case_resource_identities(
    payload: Mapping[str, Any],
    *,
    evaluator_spec: str,
    evaluator_identity: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, Any], ...]:


    resolved_identity = (
        dict(evaluator_identity)
        if evaluator_identity is not None
        else _component_code_identity(evaluator_spec, role="proposal evaluator")
    )
    if _factory_target(resolved_identity) != _CASE_DESIGN_FACTORY_TARGET:
        return ()
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise NativeCLIError(
            "case_design execution identity requires a non-empty input case_id"
        )
    raw_case_runtime = payload.get("native_case_runtime", {})
    if not isinstance(raw_case_runtime, Mapping):
        raise NativeCLIError("native_case_runtime must be a mapping")
    raw_memory_policy = raw_case_runtime.get("memory_policy", {})
    if not isinstance(raw_memory_policy, Mapping):
        raise NativeCLIError("native_case_runtime memory_policy must be a mapping")
    raw_adaptive_mode = raw_memory_policy.get("adaptive_prior_mode", "off")
    if (
        isinstance(raw_adaptive_mode, str)
        and raw_adaptive_mode.strip().lower() == "winner_commit"
    ):
        raise NativeCLIError(
            "case_design native CLI cannot safely resume winner_commit adaptive "
            "memory until ledger commits bind and verify each external memory "
            "transition"
        )
    try:
        case = resolve_case(
            case_id.strip(),
            case_root=payload.get("case_root"),
            manifest_path=payload.get("case_manifest") or payload.get("manifest_path"),
        )
    except Exception as exc:
        raise NativeCLIError(
            f"cannot resolve case {case_id!r} for execution identity"
        ) from exc
    if case.manifest_path is None:
        raise NativeCLIError(
            f"case {case_id!r} has no manifest to bind into execution identity"
        )
    manifest_path = Path(case.manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeCLIError(
            f"cannot read case manifest for execution identity: {case_id!r}"
        ) from exc
    if not isinstance(manifest, dict):
        raise NativeCLIError("case manifest must contain a JSON object")

    resources = [
        _identity_resource(manifest_path, role="case_manifest", required=True),
        _identity_resource(
            Path(case.design_state_path), role="case_design_state", required=True
        ),
        _identity_resource(
            Path(case.memory_path), role="case_adaptive_memory", required=False
        ),
    ]
    case_root = manifest_path.parent
    for field, role in (
        ("entry_program", "case_entry_program"),
        ("config_path", "case_runtime_config"),
        ("test_evaluator_path", "case_test_evaluator"),
    ):
        raw_path = manifest.get(field)
        if raw_path is None:
            continue
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise NativeCLIError(f"case manifest {field} must be a path string")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = case_root / candidate
        resources.append(_identity_resource(candidate, role=role, required=True))

    try:
        design_state = json.loads(
            Path(case.design_state_path).read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeCLIError(
            "cannot read case design state for resource identity"
        ) from exc
    if not isinstance(design_state, dict):
        raise NativeCLIError("case design state must contain a JSON object")
    raw_sheet = design_state.get("case_sheet_path")
    if raw_sheet is not None and (
        not isinstance(raw_sheet, str) or not raw_sheet.strip()
    ):
        raise NativeCLIError("case_sheet_path must be a path string")
    sheet_path = Path(raw_sheet) if raw_sheet else Path("case_sheet.json")
    if not sheet_path.is_absolute():
        sheet_path = Path(case.design_state_path).parent / sheet_path
    resources.append(
        _identity_resource(
            sheet_path,
            role="case_sheet",
            required=(raw_sheet is not None),
        )
    )
    return tuple(resources)


def _component_code_identity(spec: str, *, role: str) -> dict[str, Any]:
    try:
        return factory_code_identity(spec)
    except ExecutionIdentityError as exc:
        raise NativeCLIError(
            f"cannot establish stable {role} factory code identity"
        ) from exc


def _factory_target(identity: Mapping[str, Any]) -> tuple[Any, Any]:
    return identity.get("module"), identity.get("qualname")


def _project_source_identity() -> dict[str, Any]:
    try:
        return python_source_tree_identity(
            _PROJECT_PYTHON_SOURCE_ROOTS,
            role="dastevolve_local_python_dependencies",
        )
    except ExecutionIdentityError as exc:
        raise NativeCLIError(
            "cannot establish local Python dependency source-tree identity"
        ) from exc


def _execution_contract(
    args: argparse.Namespace,
    *,
    payload: Mapping[str, Any],
    archive_config: ArchiveProjectionConfig,
    cache_identity: EvaluationIdentityPolicy,
    cache_failure: FailureCachePolicy,
    content_determinism: Optional[ContentDeterminismDeclaration],
) -> dict[str, Any]:
    proposal_spec, evaluator_spec = _component_specs(args)
    assert proposal_spec is not None and evaluator_spec is not None
    proposal_identity = _component_code_identity(proposal_spec, role="proposal source")
    evaluator_identity = _component_code_identity(
        evaluator_spec, role="proposal evaluator"
    )
    component_targets = {
        _factory_target(proposal_identity),
        _factory_target(evaluator_identity),
    }
    plugin_targets = component_targets - _PROJECT_BUILTIN_FACTORY_TARGETS
    if plugin_targets and not args.identity_resource:
        raise NativeCLIError(
            "custom component factories require at least one --identity-resource "
            "covering their package and local dependencies"
        )
    case_design = _factory_target(evaluator_identity) == _CASE_DESIGN_FACTORY_TARGET
    if case_design:
        if args.executor != "process" or args.timeout is None:
            raise NativeCLIError(
                "case_design requires --executor process and a finite positive "
                "--timeout for bounded evaluator termination"
            )
        try:
            finite_timeout = float(args.timeout)
        except (TypeError, ValueError) as exc:
            raise NativeCLIError(
                "case_design timeout must be finite and positive"
            ) from exc
        if not (finite_timeout > 0.0 and finite_timeout < float("inf")):
            raise NativeCLIError("case_design timeout must be finite and positive")
        if cache_identity is EvaluationIdentityPolicy.CONTENT_DETERMINISTIC:
            raise NativeCLIError(
                "case_design does not support content_deterministic cache identity "
                "until its outputs have a normalization safety declaration"
            )
    archive_value = json.loads(
        json.dumps(asdict(archive_config), ensure_ascii=False, allow_nan=False)
    )
    resources = list(
        _case_resource_identities(
            payload,
            evaluator_spec=evaluator_spec,
            evaluator_identity=evaluator_identity,
        )
    )
    for index, raw_path in enumerate(args.identity_resource):
        resources.append(
            _identity_resource(
                Path(raw_path),
                role=f"plugin_resource:{index:04d}",
                required=True,
            )
        )
    core = {
        "schema_version": NATIVE_EXECUTION_CONTRACT_VERSION,
        "mode": "cli",
        "run_id": args.run_id,
        "root_seed": args.seed,
        "proposal_budget": args.proposal_budget,
        "components": {
            "proposal_source": {
                "factory_spec": proposal_spec,
                "code_identity": proposal_identity,
            },
            "proposal_evaluator": {
                "factory_spec": evaluator_spec,
                "code_identity": evaluator_identity,
            },
            "local_dependency_source_tree": _project_source_identity(),
        },
        "inputs": {
            "canonical_payload_sha256": canonical_payload_sha256(payload),
            "case_id": payload.get("case_id"),
            "resources": resources,
        },
        "archive": archive_value,
        "prompt_memory": {
            "recent_limit": args.recent_memory,
            "best_limit": args.best_memory,
            "objective_path": list(archive_config.objective_path),
            "objective_direction": archive_config.objective_direction,
        },
        "evaluation_execution": {
            "hard_timeout": args.executor == "process",
            "timeout_seconds": args.timeout,
        },
        "evaluation_cache": {
            "enabled": bool(args.evaluation_cache),
            "evaluator_namespace": args.evaluator_namespace,
            "identity_policy": cache_identity.value,
            "failure_policy": cache_failure.value,
            "content_determinism": (
                content_determinism.to_dict()
                if content_determinism is not None
                else None
            ),
        },
    }
    return seal_execution_contract(core)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_mapping(args.input_json, label="input snapshot")
    if _EXECUTION_CONTRACT_KEY in payload:
        raise NativeCLIError(
            f"input snapshot cannot define reserved field {_EXECUTION_CONTRACT_KEY!r}"
        )
    config = EvolutionRunConfig(
        run_id=args.run_id,
        total_generations=args.generations,
        proposal_budget=args.proposal_budget,
        root_seed=args.seed,
    )
    raw_archive = _read_mapping(args.archive_config, label="archive config")
    archive_config = (
        _archive_config(raw_archive)
        if raw_archive
        else ArchiveProjectionConfig(
            feature_dimensions=(
                FeatureDimension(
                    name="objective",
                    report_path=("normalized_score",),
                    minimum=0.0,
                    maximum=1.0,
                    bins=10,
                ),
            ),
            objective_path=(
                ("total_energy",)
                if args.proposal_evaluator == "case_design"
                else ("normalized_score",)
            ),
            objective_direction=(
                "minimize"
                if args.proposal_evaluator == "case_design"
                else "maximize"
            ),
        )
    )
    cache_identity, cache_failure, content_determinism = _cache_config(args)
    execution_contract = _execution_contract(
        args,
        payload=payload,
        archive_config=archive_config,
        cache_identity=cache_identity,
        cache_failure=cache_failure,
        content_determinism=content_determinism,
    )
    factory_payload = dict(payload)
    payload = {**payload, _EXECUTION_CONTRACT_KEY: execution_contract}
    snapshot = EvolutionInputSnapshot.freeze(payload)
    archive = ArchiveProjection(archive_config)
    memory = EvolutionMemoryProjection(
        MemoryProjectionConfig(
            scope_id=args.run_id,
            recent_limit=args.recent_memory,
            best_limit=args.best_memory,
            objective_path=archive_config.objective_path,
            objective_direction=archive_config.objective_direction,
        )
    )
    services = NativeRuntimeServices(
        input_snapshot=snapshot,
        run_config=config,
        archive=archive,
        memory=memory,
    )
    source, evaluator = _resolve_components(args, factory_payload, services)
    if args.dry_run:
        return {
            "schema_version": NATIVE_RUN_REPORT_VERSION,
            "dry_run": True,
            "run_id": args.run_id,
            "input_snapshot_hash": snapshot.snapshot_hash,
            "target_generations": config.total_generations,
            "proposal_budget": config.proposal_budget,
            "ledger": str(Path(args.ledger).expanduser().resolve()),
            "executor": args.executor,
            "execution_contract": execution_contract,
            "evaluation_cache": {
                "enabled": bool(args.evaluation_cache),
                "path": (
                    str(Path(args.evaluation_cache).expanduser().resolve())
                    if args.evaluation_cache
                    else None
                ),
                "evaluator_namespace": args.evaluator_namespace,
                "identity_policy": cache_identity.value,
                "failure_policy": cache_failure.value,
            },
        }

    cache_store = None
    cached_evaluator = None
    if args.evaluation_cache:
        cache_store = FileEvaluationCacheStore(args.evaluation_cache)
        cached_evaluator = CachedProposalEvaluator(
            evaluator,
            store=cache_store,
            evaluator_namespace=args.evaluator_namespace,
            failure_policy=cache_failure,
            identity_policy=cache_identity,
            content_determinism=content_determinism,
        )
        evaluator = cached_evaluator

    executor: InlineEvaluationExecutor | OwnedProcessEvaluationExecutor
    if args.executor == "inline":
        executor = InlineEvaluationExecutor(evaluation_timeout_seconds=args.timeout)
    else:
        executor = OwnedProcessEvaluationExecutor(
            max_workers=args.workers,
            start_method=args.start_method,
            evaluation_timeout_seconds=args.timeout,
        )
    try:
        orchestrator = EvolutionOrchestrator(
            config=config,
            input_snapshot=snapshot,
            proposal_source=source,
            evaluator=evaluator,
            ledger=FileGenerationLedger(args.ledger),
            projection=CompositeCommitProjection(archive, memory),
            executor=executor,
            execution_contract=execution_contract,
        )
        result = orchestrator.run()
    finally:
        executor.shutdown()
    report = {
        "schema_version": NATIVE_RUN_REPORT_VERSION,
        "dry_run": False,
        "run_id": args.run_id,
        "input_snapshot_hash": snapshot.snapshot_hash,
        "execution_contract": execution_contract,
        "progress": asdict(result.progress),
        "generations": [
            {
                "generation_index": item.generation_index,
                "generation_id": item.manifest.generation_id,
                "manifest_hash": item.manifest.manifest_hash,
                "commit_hash": item.commit.commit_hash,
                "logical_budget_used": item.commit.logical_budget_used,
                "succeeded": item.commit.succeeded,
                "failed": item.commit.failed,
                "recovered": item.recovered,
            }
            for item in result.generations
        ],
        "archive": asdict(archive.snapshot()),
        "memory": memory.snapshot().to_dict(),
        "evaluation_cache": {
            "enabled": cache_store is not None,
            "path": (
                str(cache_store.root.resolve()) if cache_store is not None else None
            ),
            "evaluator_namespace": args.evaluator_namespace,
            "identity_policy": cache_identity.value,
            "failure_policy": cache_failure.value,
            "record_count": None,
            "record_count_status": "not_computed",
            "metrics": (
                asdict(cached_evaluator.metrics())
                if cached_evaluator is not None and args.executor == "inline"
                else None
            ),
        },
    }
    if args.report:
        _atomic_json(Path(args.report).expanduser().resolve(), report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = run_from_args(args)
    except (
        EvolutionOrchestrationError,
        NativeCLIError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "NATIVE_EXECUTION_CONTRACT_VERSION",
    "NATIVE_RUN_REPORT_VERSION",
    "NativeCLIError",
    "NativeRuntimeServices",
    "main",
    "run_from_args",
]
