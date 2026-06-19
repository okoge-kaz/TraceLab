from __future__ import annotations

import json
import sys
import time
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as appmod  # noqa: E402
import syfi_llm_runtime as runtime  # noqa: E402


class TimeoutException(Exception):
    pass


class FakeSandbox:
    def __init__(self, sandbox_id: str, *, fail_set_timeout: bool = False) -> None:
        self.sandbox_id = sandbox_id
        self.fail_set_timeout = fail_set_timeout
        self.killed = False
        self.timeouts: list[int] = []

    def kill(self) -> None:
        self.killed = True

    def set_timeout(self, seconds: int) -> None:
        self.timeouts.append(seconds)
        if self.fail_set_timeout:
            raise TimeoutException("The sandbox was not found")


def test_e2b_executor_replaces_stale_sandbox_and_retries_tool():
    old = FakeSandbox("old")
    new = FakeSandbox("new")
    future: Future = Future()
    future.set_result(old)
    events = []
    recoveries = []
    calls = []

    def logger(label, value):
        events.append((label, value))

    def fake_reset_output_dir(sandbox):
        calls.append(("reset", sandbox.sandbox_id))

    def fake_run_python_tool(sandbox, code, *, timeout, max_artifact_inline_bytes):
        calls.append(("run", sandbox.sandbox_id, code, timeout, max_artifact_inline_bytes))
        if sandbox is old:
            raise TimeoutException("The sandbox was not found: likely due to sandbox timeout")
        return {"stdout": ["ok"], "stderr": [], "error": None, "results": [], "artifacts": []}

    def recover(stale):
        recoveries.append(stale)
        return new

    original_reset_output_dir = runtime.reset_output_dir
    original_run_python_tool = runtime.run_python_tool
    try:
        runtime.reset_output_dir = fake_reset_output_dir
        runtime.run_python_tool = fake_run_python_tool

        executor = runtime.E2BPythonExecutor(
            template="template",
            sandbox_timeout=600,
            allow_internet=False,
            sandbox_future=future,
            sandbox_recovery=recover,
            kill_sandbox=False,
        )
        event = executor.execute_tool_call(
            {"id": "call-1", "function": {"name": "run_python", "arguments": json.dumps({"code": "print(1)"})}},
            print_code=False,
            tool_timeout=12,
            max_artifact_inline_bytes=34,
            logger=logger,
        )
    finally:
        runtime.reset_output_dir = original_reset_output_dir
        runtime.run_python_tool = original_run_python_tool

    assert event["result"]["stdout"] == ["ok"]
    assert recoveries == [old]
    assert calls == [
        ("reset", "old"),
        ("run", "old", "print(1)", 12, 34),
        ("reset", "new"),
        ("run", "new", "print(1)", 12, 34),
    ]
    assert any(label == "e2b" and value.get("status") == "sandbox_expired" for label, value in events)
    assert executor.sandbox is new


def test_stale_sandbox_detection_is_narrow():
    assert runtime.is_stale_sandbox_error(TimeoutException("The sandbox was not found"))
    assert not runtime.is_stale_sandbox_error(TimeoutException("user code timed out"))


def test_pool_replaces_warm_sandbox_when_timeout_refresh_fails():
    pool = appmod.SandboxPool()
    key = ("session-1", "template", False)
    old = FakeSandbox("old", fail_set_timeout=True)
    new = FakeSandbox("new")
    future: Future = Future()
    future.set_result(old)
    record = appmod.SandboxRecord(future=future, expires_at=time.monotonic() + 30)
    record.active = 1
    events = []

    def logger(label, value):
        events.append((label, value))

    def fake_create_sandbox(*, template, sandbox_timeout, allow_internet):
        assert (template, sandbox_timeout, allow_internet) == ("template", 600, False)
        return new

    original_create_sandbox = appmod.runtime.create_sandbox
    try:
        appmod.runtime.create_sandbox = fake_create_sandbox
        with pool._lock:
            pool._records[key] = record

        replacement = pool._refresh_checked_out_sandbox(
            key,
            touch_future=future,
            template="template",
            sandbox_timeout=600,
            allow_internet=False,
            ttk_seconds=180,
            logger=logger,
        )

        assert replacement.result(timeout=2) is new
        assert old.killed
        with pool._lock:
            assert pool._records[key].future is replacement
            assert pool._records[key].active == 1
        assert any(
            label == "e2b" and value.get("status") == "sandbox_timeout_refresh_failed_replaced"
            for label, value in events
        )
    finally:
        appmod.runtime.create_sandbox = original_create_sandbox
        pool.close()


def test_round_log_keeps_llm_and_tool_timings():
    rounds = appmod._rounds_from_events(
        [
            (
                "llm_call",
                {
                    "id": "gen-1",
                    "model": "model",
                    "attempt": 1,
                    "turn_elapsed_ms": 1200,
                    "usage": {"completion_tokens": 20},
                    "perf": {
                        "wall_ms": 1111.1,
                        "latency_ms": 1000,
                        "generation_time_ms": 900,
                        "provider_name": "provider",
                    },
                },
            ),
            (
                "model_turn",
                {
                    "turn": 1,
                    "finish_reason": "tool_calls",
                    "usage": {"total_tokens": 100},
                    "turn_elapsed_ms": 1300,
                    "tool_calls": [{"id": "call-1", "name": "run_python"}],
                },
            ),
            (
                "tool_result",
                {
                    "tool_call_id": "call-1",
                    "duration_ms": 456,
                    "turn_elapsed_ms": 1800,
                    "summary": {"stdout": ["ok"], "stderr": [], "error": None, "results": [], "artifacts": []},
                },
            ),
        ],
        log_content=False,
    )

    assert rounds == [
        {
            "round": 1,
            "finish_reason": "tool_calls",
            "usage": {"total_tokens": 100},
            "turn_elapsed_ms": 1300,
            "llm_calls": [
                {
                    "id": "gen-1",
                    "model": "model",
                    "attempt": 1,
                    "turn_elapsed_ms": 1200,
                    "wall_ms": 1111.1,
                    "latency_ms": 1000,
                    "generation_time_ms": 900,
                    "completion_tokens": 20,
                    "provider_name": "provider",
                }
            ],
            "tools": [
                {
                    "name": "run_python",
                    "result": {
                        "ok": True,
                        "stdout_lines": 1,
                        "stderr_lines": 0,
                        "error": False,
                        "result_count": 0,
                        "artifact_count": 0,
                    },
                    "duration_ms": 456,
                    "turn_elapsed_ms": 1800,
                }
            ],
        }
    ]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
