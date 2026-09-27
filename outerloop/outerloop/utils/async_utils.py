

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_in_executor(f: Callable) -> Callable:


    @functools.wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return wrapper


async def run_with_timeout(
    coro: Callable, timeout: float, *args: Any, timeout_error_value: Any = None, **kwargs: Any
) -> Any:

    if timeout_error_value is None:
        timeout_error_value = {"error": 0.0, "timeout": True}

    try:
        return await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return timeout_error_value


async def run_sync_with_timeout(
    func: Callable, timeout: float, *args: Any, timeout_error_value: Any = None, **kwargs: Any
) -> Any:

    if timeout_error_value is None:
        timeout_error_value = {"error": 0.0, "timeout": True}

    try:
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Sync operation timed out after {timeout}s")
        return timeout_error_value


async def gather_with_concurrency(
    n: int, *tasks: asyncio.Future, return_exceptions: bool = False
) -> List[Any]:

    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: asyncio.Future) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(
        *(sem_task(task) for task in tasks), return_exceptions=return_exceptions
    )


async def retry_async(
    coro: Callable,
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Exception, tuple] = Exception,
    **kwargs: Any,
) -> Any:

    last_exception = None
    current_delay = delay

    for i in range(retries + 1):
        try:
            return await coro(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if i < retries:
                logger.warning(
                    f"Retry {i+1}/{retries} failed with {type(e).__name__}: {str(e)}. "
                    f"Retrying in {current_delay:.2f}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"All {retries+1} attempts failed. Last error: {type(e).__name__}: {str(e)}"
                )

    if last_exception:
        raise last_exception

    return None


class TaskPool:


    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.tasks: List[asyncio.Task] = []

    @property
    def semaphore(self) -> asyncio.Semaphore:

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def run(self, coro: Callable, *args: Any, **kwargs: Any) -> Any:

        async with self.semaphore:
            return await coro(*args, **kwargs)

    def create_task(self, coro: Callable, *args: Any, **kwargs: Any) -> asyncio.Task:

        task = asyncio.create_task(self.run(coro, *args, **kwargs))
        self.tasks.append(task)
        task.add_done_callback(lambda t: self.tasks.remove(t))
        return task

    async def wait_all(self) -> None:

        if self.tasks:
            await asyncio.gather(*self.tasks)

    async def cancel_all(self) -> None:

        for task in self.tasks:
            task.cancel()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
