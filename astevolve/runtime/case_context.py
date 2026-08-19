

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from astevolve.cases.base import DesignCase
from astevolve.cases.registry import resolve_case
from astevolve.domain import RunContext


CasePath = str | PathLike[str] | Path


def get_current_case(
    case_id: Optional[str] = None,
    *,
    case_root: CasePath | None = None,
    manifest_path: CasePath | None = None,
) -> DesignCase:


    return resolve_case(
        case_id,
        case_root=case_root,
        manifest_path=manifest_path,
    )


def current_case_kwargs(
    case_id: Optional[str] = None,
    *,
    case_root: CasePath | None = None,
    manifest_path: CasePath | None = None,
) -> Dict[str, str]:


    return get_current_case(
        case_id,
        case_root=case_root,
        manifest_path=manifest_path,
    ).as_kwargs()


def current_run_context(
    case_id: Optional[str] = None,
    *,
    case_root: CasePath | None = None,
    manifest_path: CasePath | None = None,
    seed: Optional[int] = None,
    settings: Optional[Mapping[str, Any]] = None,
    run_id: str = "",
) -> RunContext:


    case = get_current_case(
        case_id,
        case_root=case_root,
        manifest_path=manifest_path,
    )
    return RunContext(
        case_id=case.case_id,
        project_root=case.root,
        output_root=case.output_root,
        design_state_path=case.design_state_path,
        memory_path=case.memory_path,
        seed=seed,
        settings=dict(settings or {}),
        run_id=run_id,
    )
