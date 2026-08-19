

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from engine.history_lifecycle import DuplicateEffectiveContractError


CANDIDATE_REPAIR_SUMMARY_VERSION = "outerloop.candidate_repair_summary.v1"
DUPLICATE_PROPOSAL_REJECTION_VERSION = (
    "outerloop.duplicate_proposal_rejection.v1"
)
_DUPLICATE_OUTCOMES = frozenset(
    {"duplicate_pending", "duplicate_completed", "duplicate_failed"}
)
_DUPLICATE_STATUSES = frozenset({"pending", "completed", "failed"})
_MAX_COMPATIBLE_LLM_SEED = 2_147_483_647


class CandidateValidationError(ValueError):


    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = str(code or "candidate_validation_error")
        self.message = str(message or self.code)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


def proposal_generation_seed(
    base_seed: Optional[int],
    *,
    attempt_index: int,
) -> Optional[int]:


    if base_seed is None:
        return None
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer or None")
    if (
        isinstance(attempt_index, bool)
        or not isinstance(attempt_index, int)
        or attempt_index < 0
    ):
        raise ValueError("attempt_index must be a non-negative integer")
    mixed = (
        int(base_seed) % _MAX_COMPATIBLE_LLM_SEED + int(attempt_index)
    ) % _MAX_COMPATIBLE_LLM_SEED


    return mixed or _MAX_COMPATIBLE_LLM_SEED


def configured_proposal_seed(llm_config: Any) -> Optional[int]:


    if llm_config is None:
        return None
    models = getattr(llm_config, "models", None)
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        for model in models:
            value = getattr(model, "random_seed", None)
            if value is not None:
                return value
    return getattr(llm_config, "random_seed", None)


def is_duplicate_code_error(error: BaseException) -> bool:


    return str(getattr(error, "code", "") or "") == "duplicate_code_proposal"


def is_repairable_proposal_error(error: BaseException) -> bool:


    return is_duplicate_code_error(error) or isinstance(
        error, DuplicateEffectiveContractError
    )


def proposal_error_code(error: BaseException) -> str:


    if isinstance(error, DuplicateEffectiveContractError):
        return "duplicate_effective_contract"
    return str(getattr(error, "code", None) or type(error).__name__)


def duplicate_proposal_rejection(error: BaseException) -> Dict[str, Any]:


    if is_duplicate_code_error(error):
        category = "code_admission"
        raw = getattr(error, "artifact", None)
    elif isinstance(error, DuplicateEffectiveContractError):
        category = "effective_contract_admission"
        raw = error.artifact
    else:
        raise TypeError("error is not a repairable duplicate proposal")
    artifact = raw if isinstance(raw, Mapping) else {}
    outcome = str(artifact.get("outcome") or "")
    if outcome not in _DUPLICATE_OUTCOMES:
        outcome = "duplicate"
    status = str(artifact.get("status") or "")
    if status not in _DUPLICATE_STATUSES:
        status = None
    return {
        "schema_version": DUPLICATE_PROPOSAL_REJECTION_VERSION,
        "category": category,


        "code": proposal_error_code(error),
        "error_code": proposal_error_code(error),
        "outcome": outcome,
        "status": status,
        "contains_code_prompt_response_or_hash": False,
    }


def proposal_repair_instruction(
    error: BaseException,
    *,
    diff_based: bool,
) -> str:


    response_contract = (
        "Reply ONLY with complete literal SEARCH/REPLACE blocks and no "
        "Markdown fences or prose."
        if diff_based
        else "Reply ONLY with the complete rewritten program and no Markdown fences."
    )
    if isinstance(error, DuplicateEffectiveContractError):
        return (
            "DUPLICATE EFFECTIVE CONTRACT REPAIR REQUIRED. The previous "
            "proposal compiled to a search contract that was already admitted. "
            "Regenerate from the unchanged parent and make a material change to "
            "an executable strategy field; formatting, comments, or equivalent "
            f"aliases are insufficient. {response_contract}"
        )
    if is_duplicate_code_error(error):
        return (
            "DUPLICATE CODE REPAIR REQUIRED. The previous exact candidate source "
            "was already admitted. Regenerate from the unchanged parent with a "
            f"materially different executable proposal. {response_contract}"
        )
    if isinstance(error, CandidateValidationError):
        return (
            "CANDIDATE SEMANTIC REPAIR REQUIRED. The previous proposal failed "
            "deterministic pre-evaluator validation with "
            f"{error.code}: {error.message}. Regenerate from the unchanged "
            "parent. Every functional objective node must retain at least one "
            "active mapping action; do not disable or remove its only executable "
            f"action. {response_contract}"
        )
    code = str(getattr(error, "code", None) or type(error).__name__)
    message = str(getattr(error, "message", None) or str(error))
    return (
        "FORMAT REPAIR REQUIRED. The previous proposal was rejected with "
        f"{code}: {message}. Regenerate the proposal from the unchanged parent. "
        f"{response_contract}"
    )


def repair_attempt(error: Exception, attempt_index: int) -> Dict[str, Any]:


    if isinstance(error, DuplicateEffectiveContractError):
        category = "effective_contract_admission"
    elif is_duplicate_code_error(error):
        category = "code_admission"
    elif isinstance(error, CandidateValidationError):
        category = "semantic_validation"
    elif str(getattr(error, "code", "")).startswith("structured_"):
        category = "structured_proposal"
    else:
        category = "diff_format"
    return {
        "attempt_index": int(attempt_index),
        "category": category,
        "error_code": proposal_error_code(error),
    }


def summarize_repairs(attempts: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:


    compact = [
        {
            "attempt_index": int(item.get("attempt_index", 0) or 0),
            "category": str(item.get("category") or "unknown"),
            "error_code": str(item.get("error_code") or "unknown"),
        }
        for item in attempts
    ]
    categories: Dict[str, int] = {}
    error_codes: Dict[str, int] = {}
    for item in compact:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
        error_codes[item["error_code"]] = error_codes.get(item["error_code"], 0) + 1
    return {
        "schema_version": CANDIDATE_REPAIR_SUMMARY_VERSION,
        "rejected_attempt_count": len(compact),
        "category_counts": dict(sorted(categories.items())),
        "error_code_counts": dict(sorted(error_codes.items())),
        "attempts": compact,
        "contains_prompt_or_response": False,
    }


__all__ = [
    "CANDIDATE_REPAIR_SUMMARY_VERSION",
    "DUPLICATE_PROPOSAL_REJECTION_VERSION",
    "CandidateValidationError",
    "configured_proposal_seed",
    "duplicate_proposal_rejection",
    "is_duplicate_code_error",
    "is_repairable_proposal_error",
    "proposal_error_code",
    "proposal_generation_seed",
    "proposal_repair_instruction",
    "repair_attempt",
    "summarize_repairs",
]
