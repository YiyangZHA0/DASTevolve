

import asyncio
import logging
import subprocess
from typing import Dict, List

from outerloop.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class ClaudeCodeLLM(LLMInterface):


    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None) or "sonnet"
        self.system_message = getattr(model_cfg, "system_message", None)
        self.max_tokens = getattr(model_cfg, "max_tokens", None) or 16000
        self.timeout = getattr(model_cfg, "timeout", None) or 300
        self.weight = getattr(model_cfg, "weight", 1.0)
        self.retries = getattr(model_cfg, "retries", None)
        if self.retries is None:
            self.retries = 3
        self.retry_delay = getattr(model_cfg, "retry_delay", None)
        if self.retry_delay is None:
            self.retry_delay = 5
        self.max_budget_usd = getattr(model_cfg, "max_budget_usd", None)
        if self.max_budget_usd is None:
            self.max_budget_usd = 1.0
        self.cwd = getattr(model_cfg, "cwd", None)
        logger.info(f"Initialized ClaudeCodeLLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        system_message = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message=system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        conversation = "\n\n".join(
            f"[{str(message.get('role') or 'user').upper()}]\n"
            f"{message.get('content', '')}"
            for message in messages
        )

        command = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--no-session-persistence",
            "--output-format",
            "text",
        ]
        if system_message:
            command.extend(["--system-prompt", system_message])

        budget = kwargs.get("max_budget_usd", self.max_budget_usd)
        command.extend(["--max-budget-usd", str(budget)])

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        loop = asyncio.get_running_loop()
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._run_cli(command, conversation, timeout),
                    ),
                    timeout=timeout + 30,
                )
            except asyncio.TimeoutError:
                if attempt >= retries:
                    logger.error(
                        "All %d Claude Code attempts failed with timeout", retries + 1
                    )
                    raise
                logger.warning(
                    "Claude Code CLI timeout on attempt %d/%d. Retrying...",
                    attempt + 1,
                    retries + 1,
                )
                await asyncio.sleep(retry_delay)
            except Exception as exc:
                if attempt >= retries:
                    logger.error(
                        "All %d Claude Code attempts failed with error: %s",
                        retries + 1,
                        exc,
                    )
                    raise
                logger.warning(
                    "Claude Code CLI error on attempt %d/%d: %s. Retrying...",
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                await asyncio.sleep(retry_delay)

        raise RuntimeError("Claude Code retry loop terminated unexpectedly")

    def _run_cli(self, command: List[str], prompt: str, timeout: int) -> str:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                input=prompt,
                timeout=timeout,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise asyncio.TimeoutError("Claude CLI subprocess timed out") from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                logger.warning("Claude CLI stderr: %s", stderr[:500])
            raise RuntimeError(
                f"Claude CLI exited with status {result.returncode}: {stderr[:500]}"
            )

        output = result.stdout.strip()
        if not output:
            raise RuntimeError(
                f"Empty response from Claude CLI. stderr: {result.stderr[:500]}"
            )
        return output


def init_claude_code_client(model_cfg):


    return ClaudeCodeLLM(model_cfg)
