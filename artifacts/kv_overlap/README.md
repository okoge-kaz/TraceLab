# kv_overlap — Tool-overlapped KV handoff & assistant-output canonicalization

**Question.** In coding-agent serving, tool execution creates a window (*tool slack*)
between the end of one LLM round and the arrival of its tool results. Can we hide
inside that window (a) the **Retain-thinking** proactive D→P delta-KV handoff of the
just-generated tokens `G_i = R_i || A_i`, and (b) the **Strip-thinking** recompute
prefill of the reasoning-stripped canonical prefix `prefill(C_i, A_i)`? What stays
exposed on the critical path, and under which tool / output / prefix conditions?

**What this is (and is not).** TraceLab's public dataset is used as a **workload
trace** only: tool slack, token lengths, and step/session structure come from real
Claude/Codex sessions, and are replayed against **open-model KV sizes**
(Qwen3-235B-A22B, GLM-5.2, Kimi-K3-via-proxy) and **hypothetical network / prefill
performance**. This is a *counterfactual simulation*. Nothing here asserts that
Claude's or Codex's actual backends use P/D disaggregation, nor reproduces their
private serving stacks.

## Notation

At round *i*'s invocation the canonical prefix is `C_i` (KV assumed resident on P in
the main scenario). D generates `G_i = R_i || A_i` (reasoning + visible/tool-call
tokens); tool results are `O_i`.

- Retain-thinking next context: `C_i || R_i || A_i || O_i` → D→P must move only `G_i`.
- Strip-thinking next context: `C_i || A_i || O_i` → `A_i`'s D-side KV is unusable
  (it attends to `R_i` at different positions), so P teacher-forces `prefill(C_i, A_i)`.

## Timing definitions (aligned with `artifacts/utils/timing.py`)

- `t_generation_end(i)` = last `reasoning|text|tool_call` event of round *i*
  (`MODEL_OUTPUT_EVENT_TYPES`).
- `t_A_ready(i)` = last **non-reasoning** output event (`text|tool_call`).
- `t_tool_ready(i)` = conservative max of (a) `max(tools[].result_at)` and (b) the
  next round's last input `tool_result` timing event. Parallel tools use the max
  completion, never the sum. Both sources are stored (`tool_ready_source_diff_ms`).
- `slack_i = max(0, t_tool_ready − t_generation_end)`;
  `canonicalization_slack_i = max(0, t_tool_ready − t_A_ready)`.

## Eligible pairs

Adjacent same-session rounds (i, i+1) where round *i* emits ≥1 tool call and round
*i+1* is tool-result-initiated, with no user message in between, valid timestamps, no
compaction (next context < 50% of prior) or context reduction (shrink > 256 tok), and
round *i+1* exists. Exclusions are accounted per reason in `outputs/excluded_pairs.csv`
and `outputs/filter_stages.csv`. Three variants are reported: **unfiltered**,
**no-suspicious-prefix-jump** (drops Codex subagent-style prefix jumps), and
**strict same-session continuity** (next prefix ∈ [prev_total−tol, prev_total+output+tol]).

## Analysis A — Retain-thinking D→P delta-KV handoff

`delta_kv_tokens_i = output_tokens_i` (TraceLab output includes reasoning). This is a
*deterministic proactive handoff*, not speculation. Transfer model:

```
rounded  = ceil(tokens / block_size) * block_size
bytes    = rounded * logical_kv_bytes_per_token * physical_transfer_multiplier
transfer = fixed_overhead_ms + 1000 * bytes / (bandwidth_GBps * 1e9)
hidden   = min(slack, transfer);  exposed = max(0, transfer − slack)
```

KV bytes/token are derived from official configs (`configs/models.yaml`): Qwen3 GQA
188 KiB/tok bf16; GLM-5.2 MLA latent 87.8 KiB/tok (+9.5 KiB unverified DSA indexer);
Kimi K3 is **not public** → explicit Kimi-K2-Thinking proxy (68.6 KiB/tok) *plus* an
architecture-independent 64–256 KiB/tok sweep. Main scenario: P-side prefix resident;
prefix-miss handled as sensitivity (exclude / full-prefix transfer / lower-tier restore).

## Analysis B — Strip-thinking assistant-output canonicalization

- Codex: `visible_output_tokens = max(0, output_tokens − reasoning_output_tokens)` (exact).
- Claude: reasoning tokens are not separated in the public trace →
  `visible_output_tokens_upper = output_tokens` (**upper-bound compute estimate**, never
  treated as exact).

**Phase 1 (trace-only):** `required_prefill_tokens_per_sec = |A| / canonicalization_slack`.
**Phase 2 (synthetic profiles / lookup table):** 2D `prefill_ms(prefix, suffix)` roofline
profiles (`configs/prefill_profiles.yaml`), replaceable by a measured B300 table via
`--prefill-table`. A single constant tokens/s is deliberately not used.

Scenarios (post-tool critical path; `Q` = P-node queue, `F` = prefill, `X` = P→D transfer):

```
S0 reactive:        T = Q + F(C, A||O) + X(A||O)
S1 shadow prefill:  T = max(0, Q + F(C,A) − slack_canon) + F(C||A, O) + X(A||O)
S2 + pre-copy:      T = max(0, Q + F(C,A) + X(A) − slack_canon) + F(C||A, O) + X(O)
```

`O` is `estimated_tool_append_tokens` from the next round's `newly_append_tokens`,
per-pair classified output-cached-like vs output-resend-like (tolerances follow
`llm_generation/output_append_assignment`); it includes framing/cache-boundary effects
and is *not* a raw tool-output token count. S2 pays the fixed transfer overhead twice
and can lose to S1/S0 at short slack — this is modeled, not patched away.

## Configs

- `configs/models.yaml` — KV layouts/sizes (sources pinned), block sizes, TP sweep.
- `configs/networks.yaml` — bandwidth {25,50,70,100,200,900} GB/s × overhead
  {0.1,0.5,1,2} ms; queue sweep {0,1,5,10,50,100} ms; prefix-restore tiers.
- `configs/prefill_profiles.yaml` — B300-class synthetic roofline profiles
  (optimistic/base/pessimistic) + benchmark grid for a future measured table.

## Outputs

See `outputs/` (parquet step results, summary/sensitivity CSVs, PNGs, REPORT.md,
validation_report.csv, audit_sample_50.json). `REPORT.md` answers the research
questions and states all limitations.

## Reproduce

```bash
mkdir -p trace
curl -L --fail -o trace/syfi_coding_trace.duckdb \
  https://github.com/uw-syfi/TraceLab/releases/latest/download/syfi_coding_trace.duckdb

uv run --with pandas --with pyarrow --with pyyaml --with pytest \
  python -m pytest artifacts/kv_overlap/tests -q
uv run --with pandas --with pyarrow --with pyyaml \
  python artifacts/kv_overlap/analyze.py --db trace/syfi_coding_trace.duckdb
uv run --with pandas --with pyarrow --with pyyaml \
  python artifacts/kv_overlap/plot.py
```
