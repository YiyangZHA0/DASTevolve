

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


DEFAULT_DIFF_PATTERN = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"


class CandidateDiffError(ValueError):


    def __init__(
        self,
        code: str,
        message: str,
        *,
        diff_index: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.diff_index = diff_index
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "outerloop.candidate_diff_error.v1",
            "code": self.code,
            "message": self.message,
            "diff_index": self.diff_index,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CandidateDiffApplication:


    code: str
    changes_description: Optional[str]
    code_diffs: Tuple[Tuple[str, str], ...]
    description_diffs: Tuple[Tuple[str, str], ...]


def _normalize_newlines(value: str) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _strict_evolve_spans(code: str) -> Tuple[Tuple[int, int], ...]:


    lines = code.split("\n")
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, line in enumerate(lines):
        contains_start = "EVOLVE-BLOCK-START" in line
        contains_end = "EVOLVE-BLOCK-END" in line
        stripped = line.strip()
        if contains_start and stripped != EVOLVE_START:
            raise CandidateDiffError(
                "invalid_evolve_marker",
                f"EVOLVE start marker on line {index + 1} must be a standalone marker",
                details={"line": index + 1},
            )
        if contains_end and stripped != EVOLVE_END:
            raise CandidateDiffError(
                "invalid_evolve_marker",
                f"EVOLVE end marker on line {index + 1} must be a standalone marker",
                details={"line": index + 1},
            )
        if stripped == EVOLVE_START:
            if start is not None:
                raise CandidateDiffError(
                    "nested_evolve_block",
                    f"Nested EVOLVE block starts on line {index + 1}",
                    details={"line": index + 1},
                )
            start = index
        elif stripped == EVOLVE_END:
            if start is None:
                raise CandidateDiffError(
                    "orphan_evolve_end",
                    f"EVOLVE end marker on line {index + 1} has no open block",
                    details={"line": index + 1},
                )
            spans.append((start + 1, index))
            start = None
    if start is not None:
        raise CandidateDiffError(
            "unclosed_evolve_block",
            f"EVOLVE block starting on line {start + 1} is not closed",
            details={"line": start + 1},
        )
    if not spans:
        raise CandidateDiffError(
            "missing_evolve_block",
            "Candidate code contains no valid EVOLVE block",
        )
    return tuple(spans)


def _strip_capture_delimiter_newline(value: str) -> str:
    return value[:-1] if value.endswith("\n") else value


def _strict_extract_candidate_diffs(
    diff_text: str,
    diff_pattern: str,
) -> Tuple[Tuple[str, str], ...]:
    normalized = _normalize_newlines(diff_text)
    standard_markers = ("<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE")
    for line_number, line in enumerate(normalized.split("\n"), start=1):
        for marker in standard_markers:
            if marker in line and line.strip() != marker:
                raise CandidateDiffError(
                    "invalid_diff_syntax",
                    f"Diff marker on response line {line_number} must be standalone",
                    details={"line": line_number},
                )
    try:
        pattern = re.compile(diff_pattern, re.DOTALL)
    except re.error as exc:
        raise CandidateDiffError(
            "invalid_diff_pattern",
            f"Configured diff_pattern is invalid: {exc}",
        ) from exc
    if pattern.groups != 2:
        raise CandidateDiffError(
            "invalid_diff_pattern",
            "diff_pattern must expose exactly SEARCH and REPLACE capture groups",
            details={"capture_groups": pattern.groups},
        )
    matches = list(pattern.finditer(normalized))
    if not matches:
        raise CandidateDiffError(
            "invalid_diff_syntax",
            "Response contains no complete SEARCH/REPLACE block",
        )
    cursor = 0
    out: List[Tuple[str, str]] = []

    def reject_broken_markers(fragment: str, *, diff_index: int) -> None:
        marker_like = re.search(
            r"(?mi)^[ \t]*(?:[<@#*_=-]{2,}[^\n]*\bSEARCH\b|={3,}|"
            r"[>@#*_=-]{2,}[^\n]*\b(?:REPLACE|END)\b)[ \t]*$",
            fragment,
        )
        if marker_like:
            raise CandidateDiffError(
                "invalid_diff_syntax",
                "Response contains a malformed or partially parsed diff marker",
                diff_index=diff_index,
            )

    for index, match in enumerate(matches):


        reject_broken_markers(normalized[cursor : match.start()], diff_index=index)
        search = _strip_capture_delimiter_newline(match.group(1))
        replacement = _strip_capture_delimiter_newline(match.group(2))
        if not search:
            raise CandidateDiffError(
                "empty_search",
                "SEARCH text must not be empty",
                diff_index=index,
            )
        if search == replacement:
            raise CandidateDiffError(
                "no_effect",
                "SEARCH and REPLACE are identical",
                diff_index=index,
            )
        if "EVOLVE-BLOCK-START" in replacement or "EVOLVE-BLOCK-END" in replacement:
            raise CandidateDiffError(
                "marker_injection",
                "REPLACE text may not add or modify EVOLVE markers",
                diff_index=index,
            )
        out.append((search, replacement))
        cursor = match.end()
    reject_broken_markers(normalized[cursor:], diff_index=len(matches))
    return tuple(out)


def _linewise_matches(text: str, search: str) -> List[Tuple[int, int]]:
    lines = text.split("\n")
    search_lines = search.split("\n")
    width = len(search_lines)
    return [
        (index, index + width)
        for index in range(len(lines) - width + 1)
        if lines[index : index + width] == search_lines
    ]


def _inside_evolve_span(
    match: Tuple[int, int],
    spans: Tuple[Tuple[int, int], ...],
) -> bool:
    start, end = match
    return any(start >= content_start and end <= content_end for content_start, content_end in spans)


def _touches_evolve_boundary(
    match: Tuple[int, int],
    spans: Tuple[Tuple[int, int], ...],
) -> bool:
    start, end = match
    return any(
        start < content_end + 1 and end > content_start - 1
        for content_start, content_end in spans
    )


def _validate_nonoverlap(
    edits: List[Tuple[int, int, List[str], int]],
    *,
    target: str,
) -> None:
    ordered = sorted(edits, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise CandidateDiffError(
                "overlapping_edits",
                f"Two SEARCH blocks overlap in {target}",
                diff_index=current[3],
                details={"target": target},
            )


def _apply_planned_line_edits(
    text: str,
    edits: List[Tuple[int, int, List[str], int]],
) -> str:
    lines = text.split("\n")
    for start, end, replacement, _index in sorted(edits, reverse=True):
        lines[start:end] = replacement
    return "\n".join(lines)


def apply_candidate_diffs(
    original_code: str,
    diff_text: str,
    *,
    diff_pattern: str = DEFAULT_DIFF_PATTERN,
    changes_description: Optional[str] = None,
    require_changes_description_update: bool = False,
) -> CandidateDiffApplication:


    code = _normalize_newlines(original_code)
    description = (
        _normalize_newlines(changes_description)
        if changes_description is not None
        else None
    )
    spans = _strict_evolve_spans(code)
    diff_blocks = _strict_extract_candidate_diffs(diff_text, diff_pattern)
    code_edits: List[Tuple[int, int, List[str], int]] = []
    description_edits: List[Tuple[int, int, List[str], int]] = []
    code_diffs: List[Tuple[str, str]] = []
    description_diffs: List[Tuple[str, str]] = []

    for index, (search, replacement) in enumerate(diff_blocks):
        code_matches = _linewise_matches(code, search)
        description_matches = (
            _linewise_matches(description, search) if description is not None else []
        )
        if code_matches and description_matches:
            raise CandidateDiffError(
                "ambiguous_target",
                "SEARCH matches both code and changes_description",
                diff_index=index,
            )
        if not code_matches and not description_matches:
            raise CandidateDiffError(
                "search_not_found",
                "SEARCH does not exactly match either candidate target",
                diff_index=index,
            )
        if code_matches:
            if len(code_matches) != 1:
                raise CandidateDiffError(
                    "ambiguous_search",
                    "SEARCH must match code exactly once",
                    diff_index=index,
                    details={"match_count": len(code_matches), "target": "code"},
                )
            match = code_matches[0]
            if not _inside_evolve_span(match, spans):
                error_code = (
                    "search_crosses_evolve_boundary"
                    if _touches_evolve_boundary(match, spans)
                    else "search_outside_evolve_block"
                )
                raise CandidateDiffError(
                    error_code,
                    "Code SEARCH must lie wholly inside one EVOLVE block",
                    diff_index=index,
                    details={"start_line": match[0] + 1, "end_line": match[1]},
                )
            code_edits.append((*match, replacement.split("\n"), index))
            code_diffs.append((search, replacement))
            continue

        if len(description_matches) != 1:
            raise CandidateDiffError(
                "ambiguous_search",
                "SEARCH must match changes_description exactly once",
                diff_index=index,
                details={
                    "match_count": len(description_matches),
                    "target": "changes_description",
                },
            )
        description_edits.append(
            (*description_matches[0], replacement.split("\n"), index)
        )
        description_diffs.append((search, replacement))

    _validate_nonoverlap(code_edits, target="code")
    _validate_nonoverlap(description_edits, target="changes_description")
    if not code_edits:
        raise CandidateDiffError(
            "code_not_modified",
            "Candidate must include at least one code edit inside an EVOLVE block",
        )
    if require_changes_description_update and not description_edits:
        raise CandidateDiffError(
            "description_update_required",
            "Candidate must update changes_description",
        )

    child_code = _apply_planned_line_edits(code, code_edits)
    child_description = (
        _apply_planned_line_edits(description, description_edits)
        if description is not None
        else None
    )
    if require_changes_description_update and (
        child_description is None
        or not child_description.strip()
        or child_description == description
    ):
        raise CandidateDiffError(
            "invalid_description_update",
            "changes_description update must be non-empty and observable",
        )
    if child_code == code and child_description == description:
        raise CandidateDiffError("no_effect", "Candidate diff has no observable effect")


    _strict_evolve_spans(child_code)
    return CandidateDiffApplication(
        code=child_code,
        changes_description=child_description,
        code_diffs=tuple(code_diffs),
        description_diffs=tuple(description_diffs),
    )


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:

    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
) -> str:


    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()


    diff_blocks = extract_diffs(diff_text, diff_pattern)


    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")


        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:

                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(
    diff_text: str, diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
) -> List[Tuple[str, str]]:

    normalized = diff_text.replace("\r\n", "\n").replace("\r", "\n")

    diff_blocks = re.findall(diff_pattern, normalized, re.DOTALL)
    if not diff_blocks:


        tolerant_pattern = (
            r"(?m)^[ \t]*<<<<<<<[ \t]+SEARCH[ \t]*\n"
            r"(.*?)"
            r"^[ \t]*=======[ \t]*\n"
            r"(.*?)"
            r"^[ \t]*>>>>>>>[ \t]+REPLACE[ \t]*$"
        )
        diff_blocks = re.findall(tolerant_pattern, normalized, re.DOTALL)

    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:

    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()


    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()


    return llm_response


def _format_block_lines(lines: List[str], max_line_len: int = 100, max_lines: int = 30) -> str:

    truncated = []
    for line in lines[:max_lines]:
        s = line.rstrip()
        if len(s) > max_line_len:
            s = s[: max_line_len - 3] + "..."
        truncated.append("  " + s)
    if len(lines) > max_lines:
        truncated.append(f"  ... ({len(lines) - max_lines} more lines)")
    return "\n".join(truncated) if truncated else "  (empty)"


def format_diff_summary(
    diff_blocks: List[Tuple[str, str]],
    max_line_len: int = 100,
    max_lines: int = 30,
) -> str:

    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_block = _format_block_lines(search_lines, max_line_len, max_lines)
            replace_block = _format_block_lines(replace_lines, max_line_len, max_lines)
            summary.append(f"Change {i+1}: Replace:\n{search_block}\nwith:\n{replace_block}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:

    if code1 == code2:
        return 0


    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:


    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"


def _can_apply_linewise(haystack_lines: List[str], needle_lines: List[str]) -> bool:
    if not needle_lines:
        return False

    for i in range(len(haystack_lines) - len(needle_lines) + 1):
        if haystack_lines[i : i + len(needle_lines)] == needle_lines:
            return True

    return False


def apply_diff_blocks(original_text: str, diff_blocks: List[Tuple[str, str]]) -> Tuple[str, int]:

    lines = original_text.split("\n")
    applied = 0

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        for i in range(len(lines) - len(search_lines) + 1):
            if lines[i : i + len(search_lines)] == search_lines:
                lines[i : i + len(search_lines)] = replace_lines
                applied += 1
                break

    return "\n".join(lines), applied


def split_diffs_by_target(
    diff_blocks: List[Tuple[str, str]],
    *,
    code_text: str,
    changes_description_text: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:

    code_lines = code_text.split("\n")
    desc_lines = changes_description_text.split("\n")

    code_blocks: List[Tuple[str, str]] = []
    desc_blocks: List[Tuple[str, str]] = []
    unmatched: List[Tuple[str, str]] = []

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")

        matches_code = _can_apply_linewise(code_lines, search_lines)
        matches_desc = _can_apply_linewise(desc_lines, search_lines)

        if matches_code and matches_desc:
            raise ValueError(
                "Ambiguous diff block: SEARCH matches both code and changes_description"
            )
        if matches_code:
            code_blocks.append((search_text, replace_text))
        elif matches_desc:
            desc_blocks.append((search_text, replace_text))
        else:
            unmatched.append((search_text, replace_text))

    return code_blocks, desc_blocks, unmatched
