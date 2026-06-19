#!/usr/bin/env python3
"""Run SYFI QA tasks across one or more OpenRouter models.

The benchmark task definitions live next to this script in ``syfi_qa_tasks.json``.
This runner intentionally stays in ``web/ai_infra/test`` so benchmark code and
generated result files do not leak into the app runtime path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


TEST_DIR = Path(__file__).resolve().parent
AI_INFRA_DIR = TEST_DIR.parent
REPO_ROOT = TEST_DIR.parents[2]
DEFAULT_TASKS = TEST_DIR / "syfi_qa_tasks.json"
DEFAULT_DB = REPO_ROOT / "trace" / "syfi_coding_trace.duckdb"
DEFAULT_RESULTS_DIR = TEST_DIR / "results"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPLATE = "syfi-qa-code-interpreter:latest"
MAX_CONCURRENCY = 20
REQUESTED_OPENROUTER_MODELS = [
    "nex-agi/nex-n2-pro:free",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def split_csv(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend(item.strip() for item in value.split(",") if item.strip())
    return out


def load_tasks(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data.get("tasks"), list):
        raise SystemExit(f"{path} does not contain a tasks array")
    return data


def select_tasks(task_data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(task_data["tasks"])
    ids = set(split_csv(args.task))
    difficulties = set(split_csv(args.difficulty))
    include_tests = set(split_csv(args.include_test))
    exclude_tests = set(split_csv(args.exclude_test))
    if args.executor == "local" and not args.include_artifacts:
        exclude_tests.add("artifact_generation")

    if ids:
        selected = [task for task in selected if task["id"] in ids]
        found = {task["id"] for task in selected}
        missing = sorted(ids - found)
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(missing)}")
    if difficulties:
        selected = [task for task in selected if task.get("difficulty") in difficulties]
    if include_tests:
        selected = [
            task
            for task in selected
            if include_tests.intersection(set(task.get("tests") or []))
        ]
    if exclude_tests:
        selected = [
            task
            for task in selected
            if not exclude_tests.intersection(set(task.get("tests") or []))
        ]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def parse_models(args: argparse.Namespace) -> list[str]:
    raw = split_csv(args.models)
    if not raw:
        raw = split_csv([os.environ.get("OPENROUTER_MODELS", "")])
    if not raw:
        raw = split_csv([os.environ.get("OPENROUTER_MODEL", "") or os.environ.get("OPENROUTE_MODEL", "")])
    if not raw:
        raw = list(REQUESTED_OPENROUTER_MODELS)
    if not raw:
        raise SystemExit("Pass --models model_a,model_b or set OPENROUTER_MODELS")
    return raw


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "id": task.get("id"),
        "question": task.get("question"),
        "expected": task.get("expected"),
        "grading": task.get("grading"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def latest_records_path(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    candidates = [
        path / "records.jsonl"
        for path in results_dir.iterdir()
        if path.is_dir() and (path / "records.jsonl").is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.parent.name)[-1]


def resolve_resume_paths(raw_paths: list[Path] | None, results_dir: Path) -> list[Path]:
    if not raw_paths:
        latest = latest_records_path(results_dir)
        return [] if latest is None else [latest]
    resolved: list[Path] = []
    for raw_path in raw_paths:
        path = raw_path / "records.jsonl" if raw_path.is_dir() else raw_path
        if not path.is_file():
            raise SystemExit(f"Resume records not found: {path}")
        resolved.append(path)
    return resolved


def record_matches_current_task(record: dict[str, Any], task: dict[str, Any]) -> bool:
    fingerprint = record.get("task_fingerprint")
    if isinstance(fingerprint, str):
        return fingerprint == task_fingerprint(task)
    return record.get("question") == task.get("question")


def load_existing_records(
    paths: list[Path],
    *,
    tasks_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    sources: dict[tuple[str, str], str] = {}
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                model = record.get("model")
                task_id = record.get("task_id")
                if not isinstance(model, str) or not isinstance(task_id, str):
                    continue
                task = tasks_by_id.get(task_id)
                if task is None or not record_matches_current_task(record, task):
                    continue
                records[(model, task_id)] = record
                sources[(model, task_id)] = str(path)
    return records, sources


def load_runtime(args: argparse.Namespace):
    os.environ["SYFI_LLM_PROVIDER"] = "openrouter"
    os.environ["SYFI_LLM_BASE_URL"] = args.base_url.rstrip("/")
    if args.executor == "e2b" and not os.environ.get("E2B_API_KEY") and os.environ.get("E2B_KEY"):
        os.environ["E2B_API_KEY"] = os.environ["E2B_KEY"]
    if str(AI_INFRA_DIR) not in sys.path:
        sys.path.insert(0, str(AI_INFRA_DIR))
    import syfi_llm_runtime as runtime  # noqa: PLC0415

    return runtime


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self._fh = path.open("a", encoding="utf-8")

    def write(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(value, sort_keys=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def make_logger(
    events: list[dict[str, Any]],
    *,
    quiet: bool,
    context: dict[str, Any],
    step_writer: JsonlWriter,
):
    def logger(label: str, value: Any) -> None:
        item = {"ts": utc_now_iso(), "label": label, "value": value}
        events.append(item)
        step_writer.write({**context, **item})
        if quiet:
            return
        if label == "llm_target":
            print(
                f"    target {value.get('provider')} {value.get('model')}",
                flush=True,
            )
        elif label == "model_turn":
            tool_calls = value.get("tool_calls") or []
            print(
                f"    model turn {value.get('turn')} finish={value.get('finish_reason')} "
                f"tool_calls={len(tool_calls)}",
                flush=True,
            )
        elif label == "tool_result":
            summary = value.get("summary") or {}
            err = summary.get("error")
            print(f"    tool result error={bool(err)}", flush=True)
        elif label == "final":
            print(f"    final turns={value.get('turns')}", flush=True)
        elif label == "llm_call":
            perf = value.get("perf") or {}
            print(
                "    llm call "
                f"cost={perf.get('cost')} wall_ms={perf.get('wall_ms')} "
                f"ttft_est_ms={perf.get('ttft_estimated_ms')} "
                f"tpot_est_ms={perf.get('tpot_estimated_ms')}",
                flush=True,
            )
        elif label in {"openrouter_retry", "generation_retry", "llm_failover", "e2b"}:
            print(f"    {label}: {json.dumps(value, sort_keys=True)[:600]}", flush=True)

    return logger


NUMBER_PATTERN = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!,))|\d+(?:\.\d+)?)"
NUMBER_RE = re.compile(NUMBER_PATTERN)
DURATION_RE = re.compile(
    rf"(?P<value>{NUMBER_PATTERN})\s*"
    r"(?P<unit>milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
DAY_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:\s+00:00:00)?$")


def answer_numbers(text: str, *, include_duration_seconds: bool = False) -> list[float]:
    numbers: list[float] = []
    for match in NUMBER_RE.finditer(text):
        try:
            numbers.append(float(match.group(0).replace(",", "")))
        except ValueError:
            pass
    if include_duration_seconds:
        for match in DURATION_RE.finditer(text):
            try:
                value = float(match.group("value").replace(",", ""))
            except ValueError:
                continue
            unit = match.group("unit").lower()
            if unit in {"millisecond", "milliseconds", "msec", "msecs", "ms"}:
                numbers.append(value / 1000.0)
            elif unit in {"second", "seconds", "sec", "secs", "s"}:
                numbers.append(value)
            elif unit in {"minute", "minutes", "min", "mins"}:
                numbers.append(value * 60.0)
            elif unit in {"hour", "hours", "hr", "hrs"}:
                numbers.append(value * 3600.0)
            elif unit in {"day", "days"}:
                numbers.append(value * 86400.0)
    return numbers


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def cell_matches(
    cell: Any,
    answer: str,
    numbers: list[float],
    tolerance: float | None,
    *,
    integer_tolerance: float | None = None,
) -> bool:
    answer_norm = normalize_text(answer)
    if cell is None:
        return any(token in answer_norm for token in ("null", "none", "n/a", "not tracked", "not reported"))
    if isinstance(cell, bool):
        return str(cell).lower() in answer_norm
    if isinstance(cell, int) and not isinstance(cell, bool):
        tol = 0.0 if integer_tolerance is None else integer_tolerance
        return any(math.isclose(value, float(cell), rel_tol=0.0, abs_tol=tol) for value in numbers)
    if isinstance(cell, float):
        tol = 0.0 if tolerance is None else tolerance
        return any(math.isclose(value, cell, rel_tol=0.0, abs_tol=tol) for value in numbers)
    cell_text = str(cell)
    cell_norm = normalize_text(cell_text)
    if DAY_PREFIX_RE.match(cell_text):
        return cell_text[:10] in answer
    return cell_norm in answer_norm


def expected_cell_specs(task: dict[str, Any]) -> list[tuple[str | None, Any]]:
    expected = task.get("expected") or {}
    columns = expected.get("columns") or []
    cells: list[tuple[str | None, Any]] = []
    rows = expected.get("rows") or []
    for row in rows:
        if isinstance(row, list):
            for index, cell in enumerate(row):
                column = columns[index] if index < len(columns) and isinstance(columns[index], str) else None
                cells.append((column, cell))
    return cells


def expected_cells(task: dict[str, Any]) -> list[Any]:
    return [cell for _column, cell in expected_cell_specs(task)]


def artifact_paths(result: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for event in result.get("tool_events") or []:
        tool_result = event.get("result") or {}
        for artifact in tool_result.get("artifacts") or []:
            path = artifact.get("path")
            if isinstance(path, str):
                paths.append(path)
    return paths


def artifact_matches(required: str, paths: list[str]) -> bool:
    required_name = Path(required).name
    return any(path == required or path.endswith(required) or Path(path).name == required_name for path in paths)


def grade_task(
    task: dict[str, Any],
    result: dict[str, Any] | None,
    error: str | None,
    *,
    strict_tool_errors: bool = False,
) -> dict[str, Any]:
    grading = task.get("grading") or {}
    expected = task.get("expected") or {}
    answer = "" if result is None else str(result.get("content") or "")
    numbers = answer_numbers(answer, include_duration_seconds=bool(grading.get("duration_unit_seconds")))
    cell_specs = expected_cell_specs(task)
    tolerance = grading.get("decimal_tolerance")
    column_tolerances = grading.get("column_tolerances") or {}
    integer_tolerances = grading.get("integer_tolerances") or {}

    def cell_tolerance(column: str | None) -> float | None:
        if column is not None and column in column_tolerances:
            return float(column_tolerances[column])
        return tolerance

    def cell_integer_tolerance(column: str | None) -> float | None:
        if column is not None and column in integer_tolerances:
            return float(integer_tolerances[column])
        return grading.get("integer_tolerance")

    matched_cells = sum(
        1
        for column, cell in cell_specs
        if cell_matches(
            cell,
            answer,
            numbers,
            cell_tolerance(column),
            integer_tolerance=cell_integer_tolerance(column),
        )
    )
    paths = [] if result is None else artifact_paths(result)
    required_artifact = expected.get("required_artifact")
    artifact_ok = True if not required_artifact else artifact_matches(str(required_artifact), paths)
    tool_called = bool(result and result.get("tool_events"))
    tool_errors = [
        event.get("result", {}).get("error")
        for event in (result or {}).get("tool_events", [])
        if event.get("result", {}).get("error")
    ]
    must_call_tool = bool(grading.get("must_call_tool"))
    tool_required_ok = tool_called or not must_call_tool
    tool_error_ok = not tool_errors or not strict_tool_errors
    tool_ok = tool_required_ok and tool_error_ok
    values_ok = matched_cells == len(cell_specs)
    passed = error is None and tool_ok and values_ok and artifact_ok
    warnings = []
    if tool_errors and not strict_tool_errors:
        warnings.append("tool_errors_ignored_after_final_answer_grading")
    return {
        "passed": passed,
        "tool_called": tool_called,
        "tool_errors": tool_errors,
        "tool_error_count": len(tool_errors),
        "tool_errors_are_warnings": bool(tool_errors and not strict_tool_errors),
        "strict_tool_errors": strict_tool_errors,
        "tool_required_ok": tool_required_ok,
        "tool_ok": tool_ok,
        "expected_cells": len(cell_specs),
        "matched_cells": matched_cells,
        "values_ok": values_ok,
        "required_artifact": required_artifact,
        "artifact_paths": paths,
        "artifact_ok": artifact_ok,
        "warnings": warnings,
    }


def compact_tool_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for event in result.get("tool_events") or []:
        raw_result = event.get("result") or {}
        compact.append(
            {
                "tool_call_id": event.get("tool_call_id"),
                "name": event.get("name"),
                "code": event.get("code"),
                "stdout": raw_result.get("stdout") or [],
                "stderr": raw_result.get("stderr") or [],
                "error": raw_result.get("error"),
                "artifacts": [
                    {
                        "path": item.get("path"),
                        "size": item.get("size"),
                        "type": item.get("type"),
                        "mime": item.get("mime"),
                        "is_image": item.get("is_image"),
                        "display": item.get("display"),
                        "inline_error": item.get("inline_error"),
                    }
                    for item in raw_result.get("artifacts") or []
                ],
            }
        )
    return compact


def generation_totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    wall_ms = []
    provider_generation_ms = []
    for event in events:
        if event.get("label") != "llm_call":
            continue
        perf = event.get("value", {}).get("perf") or {}
        if perf.get("wall_ms") is not None:
            wall_ms.append(float(perf["wall_ms"]))
        if perf.get("generation_time_ms") is not None:
            provider_generation_ms.append(float(perf["generation_time_ms"]))
    return {
        "total_llm_wall_generation_seconds": round(sum(wall_ms) / 1000.0, 3) if wall_ms else None,
        "total_provider_generation_seconds": (
            round(sum(provider_generation_ms) / 1000.0, 3) if provider_generation_ms else None
        ),
    }


def make_executor(runtime, args: argparse.Namespace, *, out_dir: Path):
    if args.executor == "local":
        return runtime.LocalDuckDBExecutor(db_path=str(args.db.resolve()), out_dir=str(out_dir))
    return runtime.E2BPythonExecutor(
        template=args.template,
        sandbox_timeout=args.sandbox_timeout,
        allow_internet=args.allow_internet,
        kill_sandbox=True,
    )


def prepare_executor(runtime, executor: Any, args: argparse.Namespace, logger) -> None:
    if args.executor != "e2b":
        return
    sandbox = executor.ensure_sandbox(logger)
    runtime.reset_output_dir(sandbox)
    executor.sandbox_prepared = True


def run_one_task(
    runtime,
    *,
    args: argparse.Namespace,
    model: str,
    task: dict[str, Any],
    executor: Any,
    task_out_dir: Path,
    step_writer: JsonlWriter,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    context = {"model": model, "task_id": task["id"], "executor": args.executor}
    logger = make_logger(events, quiet=args.quiet, context=context, step_writer=step_writer)
    started = time.monotonic()
    started_at = utc_now_iso()
    logger(
        "task_start",
        {
            "difficulty": task.get("difficulty"),
            "tests": task.get("tests") or [],
            "task_out_dir": str(task_out_dir),
        },
    )
    result: dict[str, Any] | None = None
    error = None
    try:
        prepare_executor(runtime, executor, args, logger)
        result = runtime.run_chat_turn(
            messages=[{"role": "user", "content": task["question"]}],
            model=model,
            template=args.template,
            prompt_file=args.prompt_file,
            max_tool_turns=args.max_tool_turns,
            max_tokens=args.max_tokens,
            max_generation_retries=args.max_generation_retries,
            tool_timeout=args.tool_timeout,
            sandbox_timeout=args.sandbox_timeout,
            allow_internet=args.allow_internet,
            print_code=args.print_code,
            max_artifact_inline_bytes=args.max_artifact_inline_bytes,
            openrouter_max_retries=args.openrouter_max_retries,
            trace_context=runtime.DEFAULT_SYFI_TRACE_CONTEXT,
            executor=executor,
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    score = grade_task(task, result, error, strict_tool_errors=args.strict_tool_errors)
    logger(
        "task_end",
        {
            "elapsed_seconds": round(elapsed, 3),
            "error": error,
            "score": score,
        },
    )
    return {
        "suite": "syfi_qa_model_performance",
        "ts": started_at,
        "elapsed_seconds": round(elapsed, 3),
        "e2e_seconds": round(elapsed, 3),
        **generation_totals(events),
        "model": model,
        "executor": args.executor,
        "task_id": task["id"],
        "task_fingerprint": task_fingerprint(task),
        "difficulty": task.get("difficulty"),
        "tests": task.get("tests") or [],
        "question": task["question"],
        "error": error,
        "answer": "" if result is None else result.get("content"),
        "turns": None if result is None else result.get("turns"),
        "forced": None if result is None else result.get("forced"),
        "provider": None if result is None else result.get("provider"),
        "usage": None if result is None else result.get("usage"),
        "tool_events": [] if result is None else compact_tool_events(result),
        "events": events,
        "score": score,
        "task_out_dir": str(task_out_dir),
    }


def summarize(records: list[dict[str, Any]], *, task_count: int, models: list[str]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        e2e_seconds = [float(record["e2e_seconds"]) for record in model_records if record.get("e2e_seconds") is not None]
        costs = [
            float((record.get("usage") or {}).get("cost") or 0.0)
            for record in model_records
            if (record.get("usage") or {}).get("cost") is not None
        ]
        prompt_tokens = sum(int((record.get("usage") or {}).get("prompt_tokens") or 0) for record in model_records)
        completion_tokens = sum(
            int((record.get("usage") or {}).get("completion_tokens") or 0) for record in model_records
        )
        total_tokens = sum(int((record.get("usage") or {}).get("total_tokens") or 0) for record in model_records)
        llm_call_perf = [
            event.get("value", {}).get("perf") or {}
            for record in model_records
            for event in record.get("events") or []
            if event.get("label") == "llm_call"
        ]
        wall_ms = [value for perf in llm_call_perf if (value := perf.get("wall_ms")) is not None]
        generation_time_ms = [
            value for perf in llm_call_perf if (value := perf.get("generation_time_ms")) is not None
        ]
        ttft_est_ms = [
            value for perf in llm_call_perf if (value := perf.get("ttft_estimated_ms")) is not None
        ]
        tpot_est_ms = [
            value for perf in llm_call_perf if (value := perf.get("tpot_estimated_ms")) is not None
        ]
        observed_tps = [
            value
            for perf in llm_call_perf
            if (value := perf.get("observed_completion_tokens_per_second")) is not None
        ]
        by_model[model] = {
            "tasks": len(model_records),
            "passed": sum(1 for record in model_records if record["score"]["passed"]),
            "tool_called": sum(1 for record in model_records if record["score"]["tool_called"]),
            "errors": sum(1 for record in model_records if record.get("error")),
            "elapsed_seconds": round(sum(record["elapsed_seconds"] for record in model_records), 3),
            "total_e2e_seconds": round(sum(e2e_seconds), 3) if e2e_seconds else None,
            "mean_e2e_seconds": round(sum(e2e_seconds) / len(e2e_seconds), 3) if e2e_seconds else None,
            "max_e2e_seconds": round(max(e2e_seconds), 3) if e2e_seconds else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(sum(costs), 8) if costs else None,
            "llm_calls": len(llm_call_perf),
            "total_llm_wall_generation_seconds": round(sum(wall_ms) / 1000.0, 3) if wall_ms else None,
            "total_provider_generation_seconds": (
                round(sum(generation_time_ms) / 1000.0, 3) if generation_time_ms else None
            ),
            "mean_wall_ms": round(sum(wall_ms) / len(wall_ms), 3) if wall_ms else None,
            "mean_ttft_estimated_ms": round(sum(ttft_est_ms) / len(ttft_est_ms), 3) if ttft_est_ms else None,
            "mean_tpot_estimated_ms": round(sum(tpot_est_ms) / len(tpot_est_ms), 3) if tpot_est_ms else None,
            "mean_observed_completion_tokens_per_second": (
                round(sum(observed_tps) / len(observed_tps), 3) if observed_tps else None
            ),
        }

    by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        item = by_task.setdefault(record["task_id"], {"models": 0, "passed": 0})
        item["models"] += 1
        item["passed"] += int(bool(record["score"]["passed"]))

    total = len(records)
    return {
        "suite": "syfi_qa_model_performance",
        "generated_at": utc_now_iso(),
        "models": models,
        "task_count": task_count,
        "records": total,
        "passed": sum(1 for record in records if record["score"]["passed"]),
        "errors": sum(1 for record in records if record.get("error")),
        "by_model": by_model,
        "by_task": by_task,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument(
        "--models",
        action="append",
        help="Comma-separated OpenRouter model ids. Can be repeated. Env fallback: OPENROUTER_MODELS.",
    )
    parser.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--template", default=os.environ.get("E2B_SYFI_TEMPLATE", DEFAULT_TEMPLATE))
    parser.add_argument("--prompt-file", type=Path, default=AI_INFRA_DIR / "syfi_qa_system_prompt.md")
    parser.add_argument(
        "--executor",
        choices=("e2b", "local"),
        default="e2b",
        help="Use E2B for safer public-path execution, or local for trusted fast iteration.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB path for --executor local.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model/task pairs already present in prior records with a matching task fingerprint or question.",
    )
    parser.add_argument(
        "--resume-from",
        action="append",
        type=Path,
        help="Prior result directory or records.jsonl to use with --skip-existing. Can be repeated. Defaults to the latest result dir.",
    )
    parser.add_argument("--task", action="append", help="Task id or comma-separated task ids.")
    parser.add_argument("--difficulty", action="append", help="Difficulty filter: easy, medium, hard.")
    parser.add_argument("--include-test", action="append", help="Require at least one test tag.")
    parser.add_argument("--exclude-test", action="append", help="Exclude tasks with any test tag.")
    parser.add_argument("--include-artifacts", action="store_true", help="Do not auto-skip plot/artifact tasks in local mode.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected tasks after filtering.")
    parser.add_argument("--max-tool-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-generation-retries", type=int, default=3)
    parser.add_argument("--tool-timeout", type=float, default=120)
    parser.add_argument("--sandbox-timeout", type=int, default=600)
    parser.add_argument("--max-artifact-inline-bytes", type=int, default=2_000_000)
    parser.add_argument("--openrouter-max-retries", type=int, default=3)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"Maximum concurrent model/task runs. Values above {MAX_CONCURRENCY} are capped.",
    )
    parser.add_argument("--allow-internet", action="store_true", help="Allow internet inside E2B sandbox.")
    parser.add_argument("--print-code", action="store_true", help="Include tool code events in progress logs.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any task fails grading.")
    parser.add_argument(
        "--strict-tool-errors",
        action="store_true",
        help="Fail a task when any intermediate tool call errors, even if the final answer is correct.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print selected models/tasks without calling APIs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_data = load_tasks(args.tasks)
    tasks = select_tasks(task_data, args)
    models = parse_models(args)
    if not tasks:
        raise SystemExit("No tasks selected")
    tasks_by_id = {task["id"]: task for task in tasks}
    all_jobs = [(model, task) for model in models for task in tasks]
    jobs = list(all_jobs)
    inherited_records: list[dict[str, Any]] = []
    resume_paths: list[Path] = []
    skipped_existing = 0
    if args.skip_existing:
        resume_paths = resolve_resume_paths(args.resume_from, args.results_dir)
        if resume_paths:
            existing, sources = load_existing_records(resume_paths, tasks_by_id=tasks_by_id)
            jobs = []
            for model, task in all_jobs:
                pair = (model, task["id"])
                record = existing.get(pair)
                if record is None:
                    jobs.append((model, task))
                    continue
                inherited = dict(record)
                inherited["inherited"] = True
                inherited["inherited_from"] = sources[pair]
                inherited.setdefault("task_fingerprint", task_fingerprint(task))
                inherited_records.append(inherited)
            skipped_existing = len(inherited_records)

    print(f"models: {', '.join(models)}", flush=True)
    print(f"tasks: {len(tasks)}", flush=True)
    for task in tasks:
        print(f"  {task['id']} [{task.get('difficulty')}]", flush=True)
    if args.skip_existing:
        if resume_paths:
            print(
                "skip-existing: "
                f"{skipped_existing} reused, {len(jobs)} to run "
                f"from {', '.join(str(path) for path in resume_paths)}",
                flush=True,
            )
        else:
            print("skip-existing: no prior records found; running all selected pairs", flush=True)
    if args.dry_run:
        return 0

    runtime = load_runtime(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.results_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    records_path = out_dir / "records.jsonl"
    steps_path = out_dir / "steps.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"
    requested_concurrency = max(1, args.concurrency)
    concurrency = 0 if not jobs else min(MAX_CONCURRENCY, requested_concurrency, len(jobs))
    config_path.write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "concurrency": concurrency,
                "concurrency_requested": args.concurrency,
                "executor": args.executor,
                "generated_at": utc_now_iso(),
                "jobs": len(jobs),
                "models": models,
                "resume_from": [str(path) for path in resume_paths],
                "skip_existing": args.skip_existing,
                "skipped_existing": skipped_existing,
                "task_ids": [task["id"] for task in tasks],
                "template": args.template,
                "tasks_file": str(args.tasks),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    print(f"concurrency: {concurrency}", flush=True)
    step_writer = JsonlWriter(steps_path)
    print_lock = Lock()

    def run_job(job_index: int, model: str, task: dict[str, Any]) -> dict[str, Any]:
        model_out_dir = out_dir / "work" / run_slug(model)
        task_out_dir = model_out_dir / task["id"]
        task_out_dir.mkdir(parents=True, exist_ok=True)
        with print_lock:
            print(f"[{job_index}/{len(jobs)}] {model} :: {task['id']}", flush=True)
        executor = make_executor(runtime, args, out_dir=task_out_dir)
        try:
            return run_one_task(
                runtime,
                args=args,
                model=model,
                task=task,
                executor=executor,
                task_out_dir=task_out_dir,
                step_writer=step_writer,
            )
        finally:
            executor.close()

    try:
        with records_path.open("a", encoding="utf-8") as records_file:
            for record in inherited_records:
                records.append(record)
                records_file.write(json.dumps(record, sort_keys=True) + "\n")
            records_file.flush()
            if jobs:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {
                        pool.submit(run_job, index, model, task): (index, model, task)
                        for index, (model, task) in enumerate(jobs, start=1)
                    }
                    for future in as_completed(futures):
                        _index, model, task = futures[future]
                        task_id = task["id"]
                        try:
                            record = future.result()
                        except Exception as exc:  # noqa: BLE001 — keep the suite running
                            error = f"{type(exc).__name__}: {exc}"
                            record = {
                                "suite": "syfi_qa_model_performance",
                                "ts": utc_now_iso(),
                                "elapsed_seconds": 0.0,
                                "e2e_seconds": 0.0,
                                "total_llm_wall_generation_seconds": None,
                                "total_provider_generation_seconds": None,
                                "model": model,
                                "executor": args.executor,
                                "task_id": task_id,
                                "task_fingerprint": task_fingerprint(task),
                                "difficulty": task.get("difficulty"),
                                "tests": task.get("tests") or [],
                                "question": task["question"],
                                "error": error,
                                "answer": "",
                                "turns": None,
                                "forced": None,
                                "provider": None,
                                "usage": None,
                                "tool_events": [],
                                "events": [],
                                "score": grade_task(
                                    task,
                                    None,
                                    error,
                                    strict_tool_errors=args.strict_tool_errors,
                                ),
                                "task_out_dir": None,
                            }
                        records.append(record)
                        records_file.write(json.dumps(record, sort_keys=True) + "\n")
                        records_file.flush()
                        status = "PASS" if record["score"]["passed"] else "FAIL"
                        with print_lock:
                            print(
                                f"  {status} {record['model']}::{record['task_id']} "
                                f"matched={record['score']['matched_cells']}/"
                                f"{record['score']['expected_cells']} "
                                f"elapsed={record['elapsed_seconds']}s",
                                flush=True,
                            )
            if inherited_records:
                with print_lock:
                    print(f"  reused {len(inherited_records)} existing records", flush=True)
    finally:
        step_writer.close()

    summary = summarize(records, task_count=len(tasks), models=models)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nrecords: {records_path}", flush=True)
    print(f"steps: {steps_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"passed: {summary['passed']}/{summary['records']}", flush=True)
    if args.strict and summary["passed"] != summary["records"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
