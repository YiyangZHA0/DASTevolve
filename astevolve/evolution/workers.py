

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import pickle
import signal
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Tuple

from astevolve.domain import EvidenceBundle, EvidenceRecord

from .domain import Proposal, SealedEvaluation


class EvaluationOutcomeStatus(str, Enum):


    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class EvaluationOutcome:


    proposal_id: str
    slot: int
    status: EvaluationOutcomeStatus
    value: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    elapsed_seconds: float = 0.0
    worker_pid: Optional[int] = None

    @property
    def succeeded(self) -> bool:
        return self.status is EvaluationOutcomeStatus.SUCCEEDED


class EvaluationExecutorClosed(RuntimeError):
    pass


class EvaluationWorkerShutdownError(RuntimeError):
    pass


ProposalEvaluator = Callable[[Proposal], Any]


class EvaluationExecutor(Protocol):


    def evaluate_many(
        self,
        proposals: Iterable[Proposal],
        evaluator: ProposalEvaluator,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[EvaluationOutcome, ...]: ...

    def shutdown(self) -> None: ...


def _validated_timeout(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout_seconds must be a positive number or None")
    resolved = float(value)
    if resolved <= 0:
        raise ValueError("timeout_seconds must be positive")
    return resolved


def _validated_proposals(proposals: Iterable[Proposal]) -> Tuple[Proposal, ...]:
    resolved = tuple(proposals)
    for proposal in resolved:
        if not isinstance(proposal, Proposal):
            raise TypeError("proposals must contain only Proposal values")
        proposal.verify()
    return resolved


def _task_proposals(
    tasks: Iterable[Any],
) -> tuple[Tuple[Any, ...], Tuple[Proposal, ...]]:
    ordered_tasks = tuple(tasks)
    proposals = []
    for index, task in enumerate(ordered_tasks):
        proposal = getattr(task, "proposal", None)
        if not isinstance(proposal, Proposal):
            raise TypeError(f"evaluation task {index} has no valid proposal")
        proposal.verify()
        task_slot = getattr(task, "slot", proposal.slot)
        if task_slot != proposal.slot:
            raise ValueError(f"evaluation task {index} slot does not match proposal")
        proposals.append(proposal)
    return ordered_tasks, tuple(proposals)


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return text if text else repr(exc)


def _pickle_safe_value(value: Any) -> Optional[str]:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException as exc:
        return _error_text(exc)
    return None


def _failure_evaluation(
    proposal: Proposal,
    outcome: EvaluationOutcome,
    *,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> SealedEvaluation:
    resolved_type = error_type or outcome.error_type or "EvaluationWorkerFailure"
    resolved_message = (
        error_message
        or outcome.error_message
        or "evaluation worker returned no failure message"
    )
    kind = {
        EvaluationOutcomeStatus.TIMED_OUT: "evaluator_timeout",
        EvaluationOutcomeStatus.CANCELLED: "evaluator_cancelled",
    }.get(outcome.status, "evaluator_exception")
    error_fingerprint = hashlib.sha256(
        (
            "astevolve.evolution.worker_error.v1\0"
            f"{resolved_type}\0{resolved_message}"
        ).encode("utf-8", "backslashreplace")
    ).hexdigest()
    details = {
        "status": outcome.status.value,
        "error_type": resolved_type,
        "error_fingerprint": error_fingerprint,
        "elapsed_seconds": outcome.elapsed_seconds,
    }
    if outcome.worker_pid is not None:
        details["worker_pid"] = outcome.worker_pid
    evidence = EvidenceBundle.of(
        (
            EvidenceRecord(
                source="native-evolution-worker",
                kind=kind,
                available=False,
                details=details,
                warnings=("operational evaluator detail suppressed",),
            ),
        )
    )
    return SealedEvaluation.failure(
        proposal=proposal,
        reason=f"{resolved_type}: operational failure {error_fingerprint}",
        evidence=evidence,
    )


def _seal_task_outcomes(
    proposals: Tuple[Proposal, ...],
    outcomes: Tuple[EvaluationOutcome, ...],
) -> Tuple[SealedEvaluation, ...]:
    if len(proposals) != len(outcomes):
        raise RuntimeError("worker returned a different number of task outcomes")
    sealed = []
    for proposal, outcome in zip(proposals, outcomes):
        if outcome.proposal_id != proposal.proposal_id:
            raise RuntimeError("worker outcome order or proposal identity changed")
        if outcome.succeeded:
            value = outcome.value
            if isinstance(value, SealedEvaluation):
                try:
                    value.verify_for(proposal)
                except Exception as exc:
                    sealed.append(
                        _failure_evaluation(
                            proposal,
                            outcome,
                            error_type="InvalidSealedEvaluation",
                            error_message=_error_text(exc),
                        )
                    )
                else:
                    sealed.append(value)
            else:
                sealed.append(
                    _failure_evaluation(
                        proposal,
                        outcome,
                        error_type="InvalidEvaluationResult",
                        error_message="operation must return SealedEvaluation",
                    )
                )
        else:
            sealed.append(_failure_evaluation(proposal, outcome))
    return tuple(sealed)


@dataclass(frozen=True)
class _WorkerMessage:
    succeeded: bool
    value: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class _WorkerStarted:


    worker_pid: int
    process_group_id: Optional[int] = None


def _evaluate_in_child(
    sender: Connection,
    evaluator: Callable[[Any], Any],
    subject: Any,
) -> None:


    try:
        worker_pid = mp.current_process().pid
        if worker_pid is None:
            raise RuntimeError("evaluation worker has no process id")
        process_group_id: Optional[int] = None


        if hasattr(os, "setsid"):
            try:
                os.setsid()
                process_group_id = os.getpgrp()
            except OSError:
                process_group_id = None
        sender.send(
            _WorkerStarted(
                worker_pid=worker_pid,
                process_group_id=process_group_id,
            )
        )
        try:
            value = evaluator(subject)
            pickle_error = _pickle_safe_value(value)
            if pickle_error is None:
                message = _WorkerMessage(succeeded=True, value=value)
            else:
                message = _WorkerMessage(
                    succeeded=False,
                    error_type="UnpickleableEvaluationResult",
                    error_message=pickle_error,
                )
        except BaseException as exc:
            message = _WorkerMessage(
                succeeded=False,
                error_type=type(exc).__name__,
                error_message=_error_text(exc),
            )

        pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        sender.send(message)
    except BaseException:

        pass
    finally:
        sender.close()


class InlineEvaluationExecutor:


    def __init__(self, *, evaluation_timeout_seconds: Optional[float] = None) -> None:
        self._closed = False
        self._lock = threading.Lock()
        self._evaluation_timeout = _validated_timeout(evaluation_timeout_seconds)

    def evaluate_many(
        self,
        proposals: Iterable[Proposal],
        evaluator: ProposalEvaluator,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[EvaluationOutcome, ...]:
        ordered = _validated_proposals(proposals)
        timeout = _validated_timeout(timeout_seconds)
        return self._evaluate_subjects(ordered, ordered, evaluator, timeout)

    def _evaluate_subjects(
        self,
        subjects: Tuple[Any, ...],
        proposals: Tuple[Proposal, ...],
        evaluator: Callable[[Any], Any],
        timeout: Optional[float],
    ) -> Tuple[EvaluationOutcome, ...]:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        with self._lock:
            if self._closed:
                raise EvaluationExecutorClosed("evaluation executor is shut down")

        outcomes = []
        for subject, proposal in zip(subjects, proposals):
            started = time.monotonic()
            try:
                value = evaluator(subject)
                elapsed = time.monotonic() - started
                if timeout is not None and elapsed >= timeout:
                    outcome = EvaluationOutcome(
                        proposal_id=proposal.proposal_id,
                        slot=proposal.slot,
                        status=EvaluationOutcomeStatus.TIMED_OUT,
                        error_type="EvaluationTimeout",
                        error_message=(f"evaluator exceeded {timeout:.6g} seconds"),
                        elapsed_seconds=elapsed,
                    )
                else:
                    pickle_error = _pickle_safe_value(value)
                    if pickle_error is None:
                        outcome = EvaluationOutcome(
                            proposal_id=proposal.proposal_id,
                            slot=proposal.slot,
                            status=EvaluationOutcomeStatus.SUCCEEDED,
                            value=value,
                            elapsed_seconds=elapsed,
                        )
                    else:
                        outcome = EvaluationOutcome(
                            proposal_id=proposal.proposal_id,
                            slot=proposal.slot,
                            status=EvaluationOutcomeStatus.FAILED,
                            error_type="UnpickleableEvaluationResult",
                            error_message=pickle_error,
                            elapsed_seconds=elapsed,
                        )
            except BaseException as exc:
                elapsed = time.monotonic() - started
                if timeout is not None and elapsed >= timeout:
                    status = EvaluationOutcomeStatus.TIMED_OUT
                    error_type = "EvaluationTimeout"
                    error_message = f"evaluator exceeded {timeout:.6g} seconds"
                else:
                    status = EvaluationOutcomeStatus.FAILED
                    error_type = type(exc).__name__
                    error_message = _error_text(exc)
                outcome = EvaluationOutcome(
                    proposal_id=proposal.proposal_id,
                    slot=proposal.slot,
                    status=status,
                    error_type=error_type,
                    error_message=error_message,
                    elapsed_seconds=elapsed,
                )


            pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL)
            outcomes.append(outcome)
        return tuple(outcomes)

    def execute(
        self,
        tasks: Tuple[Any, ...],
        operation: Callable[[Any], SealedEvaluation],
    ) -> Tuple[SealedEvaluation, ...]:


        ordered_tasks, proposals = _task_proposals(tasks)
        outcomes = self._evaluate_subjects(
            ordered_tasks,
            proposals,
            operation,
            self._evaluation_timeout,
        )
        return _seal_task_outcomes(proposals, outcomes)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "InlineEvaluationExecutor":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.shutdown()


@dataclass
class _RunningEvaluation:
    index: int
    proposal: Proposal
    process: mp.Process
    receiver: Connection
    launched_at: float
    evaluation_started_at: Optional[float] = None
    process_group_id: Optional[int] = None


class OwnedProcessEvaluationExecutor:


    def __init__(
        self,
        *,
        max_workers: int = 1,
        start_method: str = "spawn",
        evaluation_timeout_seconds: Optional[float] = None,
        poll_interval_seconds: float = 0.005,
        startup_timeout_seconds: float = 30.0,
        shutdown_grace_seconds: float = 0.1,
        terminate_grace_seconds: float = 0.5,
        kill_grace_seconds: float = 0.5,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be a positive integer")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        available = mp.get_all_start_methods()
        if start_method not in available:
            raise ValueError(
                f"unsupported multiprocessing start method {start_method!r}; "
                f"available={available}"
            )
        for name, value in (
            ("poll_interval_seconds", poll_interval_seconds),
            ("startup_timeout_seconds", startup_timeout_seconds),
            ("shutdown_grace_seconds", shutdown_grace_seconds),
            ("terminate_grace_seconds", terminate_grace_seconds),
            ("kill_grace_seconds", kill_grace_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a non-negative number")
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative")

        self._max_workers = max_workers


        self._context: Any = mp.get_context(start_method)
        self._evaluation_timeout = _validated_timeout(evaluation_timeout_seconds)
        self._poll_interval = float(poll_interval_seconds)
        if float(startup_timeout_seconds) <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        self._startup_timeout = float(startup_timeout_seconds)
        self._shutdown_grace = float(shutdown_grace_seconds)
        self._terminate_grace = float(terminate_grace_seconds)
        self._kill_grace = float(kill_grace_seconds)
        self._state_lock = threading.RLock()
        self._evaluation_lock = threading.Lock()
        self._closed = False
        self._owned: Dict[int, mp.Process] = {}
        self._process_groups: Dict[int, int] = {}

    @property
    def start_method(self) -> str:
        return self._context.get_start_method()

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def owned_pids(self) -> Tuple[int, ...]:
        with self._state_lock:
            return tuple(sorted(self._owned))

    def _is_closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def _launch(
        self,
        index: int,
        proposal: Proposal,
        evaluator: Callable[[Any], Any],
        subject: Any,
    ) -> _RunningEvaluation:
        receiver, sender = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=_evaluate_in_child,
            args=(sender, evaluator, subject),
            name=f"astevolve-eval-{proposal.slot}",
            daemon=False,
        )
        try:


            with self._state_lock:
                if self._closed:
                    raise EvaluationExecutorClosed("evaluation executor is shut down")
                process.start()
                assert process.pid is not None
                self._owned[process.pid] = process
        except BaseException:
            receiver.close()
            sender.close()
            raise
        sender.close()
        return _RunningEvaluation(
            index=index,
            proposal=proposal,
            process=process,
            receiver=receiver,
            launched_at=time.monotonic(),
        )

    def _forget(self, process: mp.Process) -> None:
        pid = process.pid
        if pid is None:
            return
        with self._state_lock:
            self._owned.pop(pid, None)
            self._process_groups.pop(pid, None)

    def _remember_process_group(
        self,
        running: _RunningEvaluation,
        started: _WorkerStarted,
    ) -> None:
        pid = running.process.pid
        if pid is None or started.worker_pid != pid:
            raise RuntimeError("worker handshake process identity mismatch")
        group_id = started.process_group_id
        if group_id is None:
            return


        if group_id != pid or group_id <= 1:
            raise RuntimeError("worker handshake process group is not owned")
        running.process_group_id = group_id
        with self._state_lock:
            if pid in self._owned:
                self._process_groups[pid] = group_id

    @staticmethod
    def _group_alive(group_id: Optional[int]) -> bool:
        if group_id is None or not hasattr(os, "killpg"):
            return False
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _owned_group(self, process: mp.Process) -> Optional[int]:
        pid = process.pid
        if pid is None:
            return None
        with self._state_lock:
            return self._process_groups.get(pid)

    def _tree_alive(self, process: mp.Process) -> bool:
        parent_alive = bool(self._alive((process,)))
        return parent_alive or self._group_alive(self._owned_group(process))

    def _wait_trees(
        self,
        processes: Iterable[mp.Process],
        timeout: float,
    ) -> Tuple[mp.Process, ...]:
        remaining = tuple(processes)
        deadline = time.monotonic() + timeout
        while remaining:
            remaining = tuple(
                process for process in remaining if self._tree_alive(process)
            )
            delay = deadline - time.monotonic()
            if not remaining or delay <= 0:
                return remaining
            time.sleep(min(0.005, delay))
        return ()

    def _signal_tree(self, process: mp.Process, operation: str) -> None:
        group_id = self._owned_group(process)
        if group_id is not None and hasattr(os, "killpg"):
            resolved_signal = (
                signal.SIGTERM if operation == "terminate" else signal.SIGKILL
            )
            try:
                os.killpg(group_id, resolved_signal)
            except ProcessLookupError:
                pass
            return
        try:
            if process.is_alive():
                getattr(process, operation)()
        except (AssertionError, ProcessLookupError, ValueError):
            pass

    @staticmethod
    def _alive(processes: Iterable[mp.Process]) -> Tuple[mp.Process, ...]:
        alive = []
        for process in processes:
            try:
                process.join(timeout=0)
                if process.is_alive():
                    alive.append(process)
            except (AssertionError, ProcessLookupError, ValueError):
                continue
        return tuple(alive)

    @classmethod
    def _wait(
        cls,
        processes: Iterable[mp.Process],
        timeout: float,
    ) -> Tuple[mp.Process, ...]:
        alive = tuple(processes)
        deadline = time.monotonic() + timeout
        while alive:
            alive = cls._alive(alive)
            remaining = deadline - time.monotonic()
            if not alive or remaining <= 0:
                return alive
            time.sleep(min(0.005, remaining))
        return ()

    def _signal(
        self,
        processes: Iterable[mp.Process],
        operation: str,
    ) -> Tuple[mp.Process, ...]:
        signalled = []
        for process in processes:
            if not self._tree_alive(process):
                continue
            self._signal_tree(process, operation)
            signalled.append(process)
        return tuple(signalled)

    def _stop_owned_process(
        self, process: mp.Process, *, graceful: bool = True
    ) -> None:
        alive = self._wait_trees((process,), self._shutdown_grace if graceful else 0.0)
        if alive:
            alive = self._signal(alive, "terminate")
            alive = self._wait_trees(alive, self._terminate_grace)
        if alive:
            alive = self._signal(alive, "kill")
            alive = self._wait_trees(alive, self._kill_grace)
        if alive:
            raise EvaluationWorkerShutdownError(
                f"owned evaluator process survived kill: pid={process.pid}"
            )
        self._forget(process)

    def _finish(
        self,
        running: _RunningEvaluation,
        *,
        message: Optional[_WorkerMessage],
        status: Optional[EvaluationOutcomeStatus] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> EvaluationOutcome:
        elapsed = time.monotonic() - (
            running.evaluation_started_at or running.launched_at
        )
        if status is EvaluationOutcomeStatus.TIMED_OUT:


            self._stop_owned_process(running.process, graceful=False)
        else:
            self._stop_owned_process(running.process)
        running.receiver.close()

        if message is not None:
            if message.succeeded:
                outcome = EvaluationOutcome(
                    proposal_id=running.proposal.proposal_id,
                    slot=running.proposal.slot,
                    status=EvaluationOutcomeStatus.SUCCEEDED,
                    value=message.value,
                    elapsed_seconds=elapsed,
                    worker_pid=running.process.pid,
                )
            else:
                outcome = EvaluationOutcome(
                    proposal_id=running.proposal.proposal_id,
                    slot=running.proposal.slot,
                    status=EvaluationOutcomeStatus.FAILED,
                    error_type=message.error_type,
                    error_message=message.error_message,
                    elapsed_seconds=elapsed,
                    worker_pid=running.process.pid,
                )
        else:
            outcome = EvaluationOutcome(
                proposal_id=running.proposal.proposal_id,
                slot=running.proposal.slot,
                status=status or EvaluationOutcomeStatus.FAILED,
                error_type=error_type,
                error_message=error_message,
                elapsed_seconds=elapsed,
                worker_pid=running.process.pid,
            )
        pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL)
        return outcome

    def evaluate_many(
        self,
        proposals: Iterable[Proposal],
        evaluator: ProposalEvaluator,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[EvaluationOutcome, ...]:
        ordered = _validated_proposals(proposals)
        timeout = _validated_timeout(timeout_seconds)
        return self._evaluate_subjects(ordered, ordered, evaluator, timeout)

    def _evaluate_subjects(
        self,
        subjects: Tuple[Any, ...],
        proposals: Tuple[Proposal, ...],
        evaluator: Callable[[Any], Any],
        timeout: Optional[float],
    ) -> Tuple[EvaluationOutcome, ...]:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")


        try:
            pickle.dumps(evaluator, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:
            raise TypeError("evaluator must be pickleable") from exc

        with self._evaluation_lock:
            if self._is_closed():
                raise EvaluationExecutorClosed("evaluation executor is shut down")
            if len(subjects) != len(proposals):
                raise ValueError("subjects and proposals must have identical lengths")
            ordered = proposals
            if not ordered:
                return ()
            outcomes: list[Optional[EvaluationOutcome]] = [None] * len(ordered)
            active: Dict[int, _RunningEvaluation] = {}
            next_index = 0
            try:
                while next_index < len(ordered) or active:
                    while (
                        next_index < len(ordered)
                        and len(active) < self._max_workers
                        and not self._is_closed()
                    ):
                        proposal = ordered[next_index]
                        try:
                            running = self._launch(
                                next_index,
                                proposal,
                                evaluator,
                                subjects[next_index],
                            )
                        except EvaluationExecutorClosed:
                            break
                        except BaseException as exc:
                            outcomes[next_index] = EvaluationOutcome(
                                proposal_id=proposal.proposal_id,
                                slot=proposal.slot,
                                status=EvaluationOutcomeStatus.FAILED,
                                error_type="ProcessStartError",
                                error_message=_error_text(exc),
                            )
                            next_index += 1
                            continue
                        active[next_index] = running
                        next_index += 1

                    if self._is_closed() and next_index < len(ordered):
                        for index in range(next_index, len(ordered)):
                            proposal = ordered[index]
                            outcomes[index] = EvaluationOutcome(
                                proposal_id=proposal.proposal_id,
                                slot=proposal.slot,
                                status=EvaluationOutcomeStatus.CANCELLED,
                                error_type="ExecutorShutdown",
                                error_message="executor shut down before launch",
                            )
                        next_index = len(ordered)

                    completed = []
                    now = time.monotonic()
                    for index, running in tuple(active.items()):
                        message: Optional[_WorkerMessage] = None
                        try:
                            if running.receiver.poll():
                                received = running.receiver.recv()
                                if isinstance(received, _WorkerStarted):
                                    self._remember_process_group(running, received)
                                    running.evaluation_started_at = time.monotonic()
                                    received = None
                                elif not isinstance(received, _WorkerMessage):
                                    raise TypeError(
                                        "worker returned an invalid result envelope"
                                    )
                                message = received
                        except (
                            EOFError,
                            OSError,
                            TypeError,
                            pickle.PickleError,
                        ) as exc:
                            outcomes[index] = self._finish(
                                running,
                                message=None,
                                error_type="WorkerProtocolError",
                                error_message=_error_text(exc),
                            )
                            completed.append(index)
                            continue
                        if message is not None:
                            outcomes[index] = self._finish(running, message=message)
                            completed.append(index)
                            continue
                        if (
                            running.evaluation_started_at is None
                            and now - running.launched_at >= self._startup_timeout
                        ):
                            outcomes[index] = self._finish(
                                running,
                                message=None,
                                status=EvaluationOutcomeStatus.TIMED_OUT,
                                error_type="WorkerStartupTimeout",
                                error_message=(
                                    "evaluator worker did not become ready within "
                                    f"{self._startup_timeout:.6g} seconds"
                                ),
                            )
                            completed.append(index)
                            continue
                        if (
                            timeout is not None
                            and running.evaluation_started_at is not None
                            and now - running.evaluation_started_at >= timeout
                        ):
                            outcomes[index] = self._finish(
                                running,
                                message=None,
                                status=EvaluationOutcomeStatus.TIMED_OUT,
                                error_type="EvaluationTimeout",
                                error_message=(
                                    f"evaluator exceeded {timeout:.6g} seconds"
                                ),
                            )
                            completed.append(index)
                            continue
                        try:
                            alive = running.process.is_alive()
                        except (AssertionError, ValueError):
                            alive = False
                        if not alive:


                            try:
                                if running.receiver.poll():
                                    received = running.receiver.recv()
                                    if isinstance(received, _WorkerMessage):
                                        outcomes[index] = self._finish(
                                            running, message=received
                                        )
                                    elif isinstance(received, _WorkerStarted):
                                        outcomes[index] = self._finish(
                                            running,
                                            message=None,
                                            error_type="WorkerExited",
                                            error_message=(
                                                "evaluator process became ready but exited "
                                                "without a result; "
                                                f"exitcode={running.process.exitcode}"
                                            ),
                                        )
                                    else:
                                        raise TypeError("invalid worker envelope")
                                else:
                                    outcomes[index] = self._finish(
                                        running,
                                        message=None,
                                        error_type="WorkerExited",
                                        error_message=(
                                            "evaluator process exited without a result; "
                                            f"exitcode={running.process.exitcode}"
                                        ),
                                    )
                            except (EOFError, OSError, TypeError) as exc:
                                outcomes[index] = self._finish(
                                    running,
                                    message=None,
                                    error_type="WorkerProtocolError",
                                    error_message=_error_text(exc),
                                )
                            completed.append(index)

                    for index in completed:
                        active.pop(index, None)
                    if active and not completed:
                        time.sleep(self._poll_interval)
            finally:

                for running in tuple(active.values()):
                    try:
                        self._stop_owned_process(running.process)
                    finally:
                        running.receiver.close()

            assert all(outcome is not None for outcome in outcomes)
            return tuple(outcome for outcome in outcomes if outcome is not None)

    def execute(
        self,
        tasks: Tuple[Any, ...],
        operation: Callable[[Any], SealedEvaluation],
    ) -> Tuple[SealedEvaluation, ...]:


        ordered_tasks, proposals = _task_proposals(tasks)
        outcomes = self._evaluate_subjects(
            ordered_tasks,
            proposals,
            operation,
            self._evaluation_timeout,
        )
        return _seal_task_outcomes(proposals, outcomes)

    def shutdown(self) -> None:
        with self._state_lock:
            self._closed = True
            processes = tuple(self._owned.values())

        alive = self._wait_trees(processes, self._shutdown_grace)
        if alive:
            alive = self._signal(alive, "terminate")
            alive = self._wait_trees(alive, self._terminate_grace)
        if alive:
            alive = self._signal(alive, "kill")
            alive = self._wait_trees(alive, self._kill_grace)

        with self._state_lock:
            for process in processes:
                if process not in alive and process.pid is not None:
                    self._owned.pop(process.pid, None)
            remaining = tuple(
                process for process in self._owned.values() if process in alive
            )
        if remaining:
            raise EvaluationWorkerShutdownError(
                "owned evaluator processes survived kill: "
                + ", ".join(str(process.pid) for process in remaining)
            )

    def __enter__(self) -> "OwnedProcessEvaluationExecutor":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.shutdown()


__all__ = [
    "EvaluationExecutor",
    "EvaluationExecutorClosed",
    "EvaluationOutcome",
    "EvaluationOutcomeStatus",
    "EvaluationWorkerShutdownError",
    "InlineEvaluationExecutor",
    "OwnedProcessEvaluationExecutor",
    "ProposalEvaluator",
]
