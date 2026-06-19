# SYFI QA Model Tests

This directory holds a small benchmark set for the OpenRouter -> E2B -> DuckDB QA loop.

The task file is `syfi_qa_tasks.json`. Each task has:

- `id`: stable task identifier.
- `difficulty`: `easy`, `medium`, or `hard`.
- `tests`: model capabilities this task is meant to exercise.
- `question`: the only field that should be sent to the model.
- `expected`: reference answer data computed from `trace/syfi_coding_trace.duckdb`.
- `grading`: checks a future evaluator should apply.

Do not include `expected` in the model prompt. It is for scoring only.

## Task Mix

The current set has 25 tasks:

- Easy tasks check basic table counts, grouping, top-k, and simple sums.
- Medium tasks check joins, rates, null handling, percentiles, and provider splits.
- Hard tasks check deterministic ranking, session-style grouping, timestamp bucketing, multi-step
  CTEs, human-wait segmentation, prefix continuity, pricing/cost coverage, and plot/artifact
  generation.

## Batch OpenRouter Runner

`run_openrouter_benchmark.py` runs selected tasks against one or more OpenRouter model ids and writes
results under `web/ai_infra/test/results/<timestamp>/`.

If `--models` is omitted, the runner defaults to:

- `nex-agi/nex-n2-pro:free`

Pass `--models` to run a broader comparison set.

Dry-run the selection:

```bash
.venv/bin/python web/ai_infra/test/run_openrouter_benchmark.py \
  --difficulty easy \
  --dry-run
```

Run a small public-path benchmark through E2B:

```bash
E2B_API_KEY="$E2B_KEY" OPENROUTER_API_KEY="$OPENROUTE_KEY" \
  .venv/bin/python web/ai_infra/test/run_openrouter_benchmark.py \
  --difficulty easy \
  --limit 2 \
  --concurrency 6
```

This uses external services: task prompts and compact tool results go to OpenRouter, and generated
Python runs in an E2B sandbox over the public SYFI DuckDB environment.
Concurrency is capped at 20 even if a larger `--concurrency` is passed.

Resume a run while only executing missing or changed model/task pairs:

```bash
E2B_API_KEY="$E2B_KEY" OPENROUTER_API_KEY="$OPENROUTE_KEY" \
  .venv/bin/python web/ai_infra/test/run_openrouter_benchmark.py \
  --skip-existing \
  --resume-from web/ai_infra/test/results/20260613T114228Z \
  --concurrency 20
```

`--skip-existing` reuses prior records only when the model id, task id, and task fingerprint match.
For older records without fingerprints, the current task question must match. The new result
directory is still self-contained: inherited records are copied into its `records.jsonl` and marked
with `inherited_from`.

For cheaper local iteration, use the local DuckDB executor. This executes model-generated Python in
the current process, so use it only for trusted development runs:

```bash
OPENROUTER_API_KEY="$OPENROUTE_KEY" \
  .venv/bin/python web/ai_infra/test/run_openrouter_benchmark.py \
  --executor local \
  --db trace/syfi_coding_trace.duckdb \
  --models model-a,model-b \
  --task easy_01_provider_round_counts
```

Each run writes:

- `config.json`: models, task ids, executor, template, and source task file.
- `records.jsonl`: one JSON record per model/task, including final answer, tool code/stdout, usage,
  basic grading, progress events, per-task `e2e_seconds`, and total generation time fields.
- `steps.jsonl`: immediate per-step events for each model/task, including `llm_call` events with
  usage cost, token counts, OpenRouter generation stats, wall time, estimated TTFT/TPOT fields, and
  observed completion tokens/second over wall-clock LLM-call time.
- `summary.json`: aggregate pass/error/tool-use counts, cost, token counts, e2e time, total
  generation time, and speed fields by model and task.

OpenRouter non-streaming responses reliably include token counts and cost. True TTFT/TPOT require
streaming; without streaming, the runner records `wall_ms` and
`observed_completion_tokens_per_second`, and fills `ttft_estimated_ms` / `tpot_estimated_ms` only
when OpenRouter generation metadata is available after the call.
`total_llm_wall_generation_seconds` is always derived from summed LLM call wall time.
`total_provider_generation_seconds` is filled only when provider generation metadata is available.

## Manual Smoke Example

Run one task manually:

```bash
source ~/.bashrc
E2B_API_KEY="$E2B_KEY" OPENROUTER_API_KEY="$OPENROUTE_KEY" \
  .venv/bin/python web/ai_infra/syfi_llm_runtime.py \
  --template syfi-qa-code-interpreter:latest \
  --model tencent/hy3-preview \
  --question "Count SYFI rounds by provider. Return provider names and exact counts." \
  --print-code
```

The batch runner above performs that loop and applies a first-pass deterministic grader. By default,
the grader is final-answer-first: it checks required tool use, required artifacts, and whether expected
primitive values appear in the final answer within each task's decimal tolerance. Intermediate tool
errors are logged as warnings so recovered runs can pass. Add `--strict-tool-errors` to make any tool
execution error fail the task.

## Scoring Notes

At minimum, score:

- Tool use: data questions should call `run_python`.
- SQL correctness: uses real table/column names and joins child tables by `round_pk`.
- Result correctness: exact integer matches; decimal values within the task tolerance.
- Recovered tool errors: keep them in `records.jsonl` / `steps.jsonl`, but do not fail by default
  when the final answer and required artifacts are correct.
- Resource behavior: avoids full-table pandas loads unless the task explicitly requires sampling.
- Artifact behavior: plot tasks should save the requested PNG under `/out`.
