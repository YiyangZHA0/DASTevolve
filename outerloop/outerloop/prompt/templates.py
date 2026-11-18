

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on targeted changes that improve the declared primary objective in its stated direction.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""


DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

{optimizer_memory}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements that move the primary objective in its declared direction.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Return only complete SEARCH/REPLACE blocks with literal marker lines. Do not use Markdown fences or prose outside the blocks.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""


FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

{optimizer_memory}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""


EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

{inspirations_section}
"""


PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""


TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""


INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""


INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, Type: {program_type})
```{language}
{program_snippet}
```
Unique approach: {unique_features}
"""


EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""


DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "inspirations_section": INSPIRATIONS_SECTION_TEMPLATE,
    "inspiration_program": INSPIRATION_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
}


class TemplateManager:


    def __init__(self, custom_template_dir: Optional[str] = None):

        self.default_dir = Path(__file__).parent.parent / "prompts" / "defaults"
        self.custom_dir = Path(custom_template_dir) if custom_template_dir else None


        self.templates = {}
        self.fragments = {}


        self._load_from_directory(self.default_dir)


        if self.custom_dir:
            if self.custom_dir.exists():
                self._load_from_directory(self.custom_dir)
            else:
                logger.warning(
                    f"Custom template directory does not exist, using default prompt."
                )

    def _load_from_directory(self, directory: Path) -> None:

        if not directory.exists():
            return


        for txt_file in directory.glob("*.txt"):
            template_name = txt_file.stem
            with open(txt_file, "r", encoding="utf-8") as f:
                self.templates[template_name] = f.read()


        fragments_file = directory / "fragments.json"
        if fragments_file.exists():
            with open(fragments_file, "r", encoding="utf-8") as f:
                loaded_fragments = json.load(f)
                self.fragments.update(loaded_fragments)

    def get_template(self, name: str) -> str:

        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]

    def get_fragment(self, name: str, **kwargs) -> str:

        if name not in self.fragments:
            return f"[Missing fragment: {name}]"
        try:
            return self.fragments[name].format(**kwargs)
        except KeyError as e:
            return f"[Fragment formatting error: {e}]"

    def add_template(self, template_name: str, template: str) -> None:

        self.templates[template_name] = template

    def add_fragment(self, fragment_name: str, fragment: str) -> None:

        self.fragments[fragment_name] = fragment
