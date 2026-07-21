#!/usr/bin/env python3
"""Trace-driven analysis of tool-overlapped KV handoff (Retain-thinking) and
tool-overlapped assistant-output canonicalization (Strip-thinking).

Replays TraceLab workload timing (tool slack, token lengths) against open-model
KV sizes (Qwen3-235B-A22B / GLM-5.2 / Kimi-K3-proxy) and hypothetical network /
prefill performance. This is a counterfactual simulation of the *workload*; it
does not claim anything about the actual Claude/Codex serving backends.

Run:
    uv run --with pandas --with pyarrow --with pyyaml python \
        artifacts/kv_overlap/analyze.py --db trace/syfi_coding_trace.duckdb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]  # kv_overlap -> artifacts -> repo
sys.path.insert(0, str(REPO_ROOT / "artifacts" / "utils"))
sys.path.insert(0, str(EXP_DIR))

import trace_db  # noqa: E402
from timing import MODEL_OUTPUT_EVENT_TYPES  # noqa: E402  ({'reasoning','text','tool_call'})

import kv_models as km  # noqa: E402

DEFAULT_OUTPUT_DIR = EXP_DIR / "outputs"
ASSISTANT_OUTPUT_EVENT_TYPES = ("text", "tool_call")  # A_i-ready events (reasoning excluded)

# README-pinned dataset facts used by --validate.
README_ROUND_COUNT = 357_161
README_TOOL_COUNT = 432_510
# Paper Table 8 medians (tokens) and Table 7 tool-execution quantiles (seconds).
PAPER_TABLE8_MEDIANS = {
    ("claude", "prefix"): 126_180, ("codex", "prefix"): 115_584,
    ("claude", "append"): 857, ("codex", "append"): 886,
    ("claude", "output"): 252, ("codex", "output"): 184,
}
PAPER_TABLE7_TOOL_EXEC = {"per_step_p50_s": 0.1, "per_step_p90_s": 10.0,
                          "per_request_p50_s": 0.3}

COMPACTION_RATIO = 0.5          # next input total < 50% of prev context -> compaction
CONTEXT_REDUCTION_TOL = 256     # tokens of allowed shrink before context_reduction
PREFIX_JUMP_TOL = 2048          # tokens beyond explainable growth -> suspicious jump

TOOL_CATEGORIES = {
    "exec": {"exec_command", "Bash", "shell_command", "shell", "write_stdin",
             "send_input", "BashOutput", "KillShell"},
    "file_read": {"Read", "view_image", "NotebookRead"},
    "file_edit": {"apply_patch", "Edit", "Write", "MultiEdit", "NotebookEdit"},
    "search": {"Grep", "Glob", "ToolSearch", "LS"},
    "web": {"WebFetch", "WebSearch"},
    "subagent": {"Agent", "Task", "spawn_agent", "wait_agent", "close_agent",
                 "SendMessage", "TaskOutput"},
    "planning": {"TaskUpdate", "TaskCreate", "TaskList", "TaskGet", "TaskStop",
                 "TodoWrite", "update_plan", "ExitPlanMode", "EnterPlanMode",
                 "Monitor", "ScheduleWakeup", "CronCreate", "CronDelete", "CronList",
                 "Skill", "StructuredOutput"},
    "user_interaction": {"AskUserQuestion", "request_user_input"},
}


def tool_category(name: str | None) -> str:
    if not isinstance(name, str) or not name:
        return "other"
    if name.startswith("mcp__"):
        return "mcp"
    for category, names in TOOL_CATEGORIES.items():
        if name in names:
            return category
    return "other"


# ---------------------------------------------------------------------------
# 1. Round-level feature extraction (SQL)
# ---------------------------------------------------------------------------

def extract_round_features(con) -> pd.DataFrame:
    output_types = ", ".join(f"'{t}'" for t in sorted(MODEL_OUTPUT_EVENT_TYPES))
    a_types = ", ".join(f"'{t}'" for t in ASSISTANT_OUTPUT_EVENT_TYPES)
    sql = f"""
    WITH ev AS (
      SELECT round_pk,
        CAST(max(epoch_us(timestamp)) FILTER (WHERE event_type IN ({output_types})) AS BIGINT) AS gen_end_us,
        CAST(min(epoch_us(timestamp)) FILTER (WHERE event_type IN ({output_types})) AS BIGINT) AS gen_start_us,
        CAST(max(epoch_us(timestamp)) FILTER (WHERE event_type IN ({a_types})) AS BIGINT) AS a_ready_us,
        CAST(min(epoch_us(timestamp)) AS BIGINT) AS first_ev_us,
        CAST(max(epoch_us(timestamp)) AS BIGINT) AS last_ev_us,
        count(*) FILTER (WHERE event_type = 'user_message') AS user_msg_events,
        count(*) FILTER (WHERE event_type = 'tool_result') AS input_tool_result_events,
        CAST(max(epoch_us(timestamp)) FILTER (WHERE event_type = 'tool_result') AS BIGINT) AS input_tool_result_last_us
      FROM timing_events GROUP BY round_pk
    ),
    tc AS (
      SELECT round_pk,
        count(*) AS n_tools,
        count(*) FILTER (WHERE result_at IS NULL) AS n_tools_missing_result,
        CAST(max(epoch_us(result_at)) AS BIGINT) AS tools_ready_us,
        CAST(min(epoch_us(emitted_at)) AS BIGINT) AS first_emit_us,
        CAST(max(epoch_us(emitted_at)) AS BIGINT) AS last_emit_us,
        min(coalesce(tool_internal_latency_ms, tool_wall_latency_ms)) AS min_tool_eff_ms,
        max(coalesce(tool_internal_latency_ms, tool_wall_latency_ms)) AS max_tool_eff_ms,
        sum(coalesce(tool_internal_latency_ms, tool_wall_latency_ms)) AS sum_tool_eff_ms,
        arg_max(tool_name, epoch_us(result_at)) AS critical_tool_name,
        list_sort(list(DISTINCT tool_name)) AS tool_names_list,
        bool_or(coalesce(is_error, false)) AS any_tool_error
      FROM tool_calls GROUP BY round_pk
    )
    SELECT r.round_pk, r.ingest_seq, r.provider, r.project, r.session_id, r.session_file,
      r.round_index, r.round_id, r.model, CAST(r.turn_id AS VARCHAR) AS turn_id,
      r.input_tokens_total, r.prefix_tokens, r.newly_append_tokens, r.output_tokens,
      r.reasoning_output_tokens, r.current_user_message_count, r.current_tool_result_count,
      r.first_input_event_type,
      ev.gen_end_us, ev.gen_start_us, ev.a_ready_us, ev.first_ev_us, ev.last_ev_us,
      ev.user_msg_events, ev.input_tool_result_events, ev.input_tool_result_last_us,
      tc.n_tools, tc.n_tools_missing_result, tc.tools_ready_us, tc.first_emit_us,
      tc.last_emit_us, tc.min_tool_eff_ms, tc.max_tool_eff_ms, tc.sum_tool_eff_ms,
      tc.critical_tool_name, tc.tool_names_list, tc.any_tool_error
    FROM rounds r
    LEFT JOIN ev USING (round_pk)
    LEFT JOIN tc USING (round_pk)
    ORDER BY r.provider, r.project, r.session_id, r.session_file, r.round_index, r.ingest_seq
    """
    df = con.execute(sql).df()
    df["n_tools"] = df["n_tools"].fillna(0).astype(int)
    df["tool_names"] = df["tool_names_list"].map(
        lambda v: ",".join(v) if isinstance(v, (list, np.ndarray)) else ""
    )
    df = df.drop(columns=["tool_names_list"])
    df["tool_category"] = df["critical_tool_name"].map(tool_category)
    return df


# ---------------------------------------------------------------------------
# 2. Pair construction with exclusion accounting
# ---------------------------------------------------------------------------

def build_pairs(rounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adjacent same-session (i, i+1) pairs -> (eligible, excluded, filter_stages)."""
    session_key = ["provider", "project", "session_id", "session_file"]
    df = rounds.sort_values(session_key + ["round_index", "ingest_seq"]).reset_index(drop=True)
    numeric_cols = [
        "input_tokens_total", "prefix_tokens", "newly_append_tokens", "output_tokens",
        "reasoning_output_tokens", "current_user_message_count",
        "gen_end_us", "gen_start_us", "a_ready_us", "first_ev_us", "last_ev_us",
        "user_msg_events", "input_tool_result_events", "input_tool_result_last_us",
        "n_tools", "n_tools_missing_result", "tools_ready_us", "first_emit_us",
        "last_emit_us", "min_tool_eff_ms", "max_tool_eff_ms", "sum_tool_eff_ms",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grp = df.groupby(session_key, sort=False, dropna=False)

    nxt_cols = ["round_pk", "round_index", "prefix_tokens", "newly_append_tokens",
                "input_tokens_total", "user_msg_events", "current_user_message_count",
                "first_input_event_type", "input_tool_result_events",
                "input_tool_result_last_us", "gen_start_us", "gen_end_us", "first_ev_us"]
    for col in nxt_cols:
        df[f"next_{col}"] = grp[col].shift(-1)
    df["is_last_in_session"] = df["next_round_pk"].isna()

    stages: list[dict] = []
    n_total = len(df)
    stages.append({"stage": "all_rounds", "count": n_total})

    reason = pd.Series("", index=df.index, dtype=object)

    def mark(mask: pd.Series, name: str) -> None:
        mask = mask & (reason == "")
        reason[mask] = name

    has_tools = df["n_tools"] > 0
    # Session-terminal rounds: a no-tool terminal round is the session's final
    # answer (session_boundary); a tool-calling terminal round lost its
    # continuation (missing_next_round).
    mark(df["is_last_in_session"] & ~has_tools, "session_boundary")
    mark(df["is_last_in_session"] & has_tools, "missing_next_round")
    mark(~has_tools, "no_tool")

    # Tool-result timestamps: primary = round i's tools[].result_at; secondary =
    # the next round's input tool_result timing events. Keep both; main uses the
    # conservative (later) one.
    ready_result = df["tools_ready_us"]
    ready_events = df["next_input_tool_result_last_us"].where(
        df["next_input_tool_result_events"].fillna(0) > 0
    )
    df["tool_ready_result_at_us"] = ready_result
    df["tool_ready_next_events_us"] = ready_events
    df["tool_ready_us"] = np.fmax(ready_result, ready_events)
    df["tool_ready_source_diff_ms"] = (ready_events - ready_result) / 1000.0

    mark(has_tools & df["tool_ready_us"].isna(), "missing_tool_result")
    mark(
        (df["next_user_msg_events"].fillna(0) > 0)
        | (df["next_current_user_message_count"].fillna(0) > 0)
        | (df["next_first_input_event_type"] == "user_message"),
        "user_intervened",
    )
    # Next round must actually be tool-result-initiated.
    mark(df["next_first_input_event_type"].notna()
         & (df["next_first_input_event_type"] != "tool_result"), "other")

    invalid_ts = (
        df["gen_end_us"].isna()
        | df["gen_start_us"].isna()
        | (df["tool_ready_us"] < df["gen_start_us"])
        | (df["next_first_ev_us"] < df["gen_start_us"])
        | df["next_gen_end_us"].isna()
    )
    mark(invalid_ts, "invalid_timestamp")

    prev_ctx_total = df["input_tokens_total"].fillna(0) + df["output_tokens"].fillna(0)
    next_total = df["next_input_tokens_total"]
    mark(next_total.notna() & (next_total < COMPACTION_RATIO * prev_ctx_total), "compaction")
    mark(next_total.notna()
         & (next_total < df["input_tokens_total"] - CONTEXT_REDUCTION_TOL),
         "context_reduction")
    mark(next_total.isna() | (df["output_tokens"].fillna(0) <= 0)
         | (df["input_tokens_total"].fillna(0) <= 0), "other")

    df["exclusion_reason"] = reason
    eligible = df[reason == ""].copy()
    excluded = df[reason != ""][
        ["round_pk", "provider", "project", "session_id", "session_file",
         "round_index", "model", "n_tools", "exclusion_reason"]
    ].copy()

    counts = excluded["exclusion_reason"].value_counts()
    running = n_total
    order = ["session_boundary", "missing_next_round", "no_tool", "missing_tool_result",
             "user_intervened", "invalid_timestamp", "compaction", "context_reduction",
             "other"]
    for name in order:
        removed = int(counts.get(name, 0))
        running -= removed
        stages.append({"stage": f"minus_{name}", "removed": removed, "count": running})

    eligible = _derive_pair_features(eligible)
    stages.append({"stage": "eligible_unfiltered", "count": len(eligible)})
    stages.append({"stage": "eligible_no_suspicious_jump",
                   "count": int((~eligible["suspicious_prefix_jump"]).sum())})
    stages.append({"stage": "eligible_strict_continuity",
                   "count": int(eligible["strict_continuity"].sum())})
    return eligible, excluded, pd.DataFrame(stages)


def _derive_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pair_id"] = np.arange(len(df), dtype=np.int64)

    df["slack_ms"] = km.clip_slack_ms(df["tool_ready_us"], df["gen_end_us"])
    a_ready = df["a_ready_us"].fillna(df["gen_end_us"])
    df["canon_slack_ms"] = km.clip_slack_ms(df["tool_ready_us"], a_ready)

    df["visible_output_tokens"] = np.where(
        df["provider"] == "codex",
        km.codex_visible_output_tokens(df["output_tokens"], df["reasoning_output_tokens"]),
        df["output_tokens"],  # Claude: reasoning not separable -> upper bound
    )
    df["visible_is_upper_bound"] = df["provider"] != "codex"

    df["prefix_delta"] = df["next_prefix_tokens"] - df["input_tokens_total"]
    df["tool_wall_span_ms"] = (df["tool_ready_us"] - df["first_emit_us"]) / 1000.0
    df["parallel_tools"] = (df["n_tools"] > 1) & (
        df["sum_tool_eff_ms"].fillna(0) > df["tool_wall_span_ms"] * 1.05
    )
    df["response_time_ms"] = np.maximum(
        1.0, (df["next_gen_end_us"] - df["tool_ready_us"]) / 1000.0
    )

    # Suspicious prefix jumps (Codex subagent/session reassignment artifacts):
    # prefix grew more than everything that existed, or shrank below the prior
    # prefix, beyond tolerance.
    growth_cap = df["input_tokens_total"] + df["output_tokens"] + PREFIX_JUMP_TOL
    df["suspicious_prefix_jump"] = (
        (df["next_prefix_tokens"] > growth_cap)
        | (df["next_prefix_tokens"] < df["prefix_tokens"] - PREFIX_JUMP_TOL)
    )
    # Strict same-session continuity: next prefix within [prev_total - tol,
    # prev_total + output + tol] (P-resident prefix plausible) and contiguous
    # round index.
    tol = np.maximum(512.0, 0.1 * df["output_tokens"].fillna(0))
    df["strict_continuity"] = (
        (df["next_prefix_tokens"] >= df["input_tokens_total"] - tol)
        & (df["next_prefix_tokens"] <= growth_cap)
        & (df["next_round_index"] == df["round_index"] + 1)
        & ~df["suspicious_prefix_jump"]
    )

    labels = km.classify_output_assignment(
        df["prefix_delta"].to_numpy(), df["next_newly_append_tokens"].to_numpy(),
        df["output_tokens"].to_numpy(),
    )
    df["output_assignment"] = labels
    est = km.estimated_tool_append_tokens(
        labels, df["next_newly_append_tokens"].to_numpy(),
        df["visible_output_tokens"].to_numpy(),
    )
    df["estimated_tool_append_tokens"] = est
    df["est_append_valid"] = np.isfinite(est) & (est >= 0)
    return df


# ---------------------------------------------------------------------------
# 3. Retain-thinking analysis
# ---------------------------------------------------------------------------

def _wq(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> list[float]:
    """Weighted quantiles (linear on the weighted CDF)."""
    if len(values) == 0:
        return [float("nan")] * len(qs)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return [float("nan")] * len(qs)
    cdf = (cw - 0.5 * w) / cw[-1]
    return [float(np.interp(q, cdf, v)) for q in qs]


def _retain_metrics(tokens: np.ndarray, slack_ms: np.ndarray, *, bpt: float,
                    block: int, bw: float, oh: float, mult: float = 1.0) -> dict:
    t_ms = km.transfer_ms(tokens, bytes_per_token=bpt, block_size_tokens=block,
                          bandwidth_gbps=bw, fixed_overhead_ms=oh,
                          physical_transfer_multiplier=mult)
    tbytes = km.transfer_bytes(tokens, bpt, block, mult)
    hidden, exposed, full, frac = km.overlap_split(t_ms, slack_ms)
    n = len(t_ms)
    sp = _wq(exposed, np.ones(n), [0.5, 0.9, 0.99])
    bp = _wq(exposed, tbytes, [0.5, 0.9, 0.99])
    return {
        "n_steps": n,
        "fully_hidden_frac": float(np.mean(full)) if n else float("nan"),
        "byte_weighted_hidden_frac": float(np.sum(hidden / np.maximum(t_ms, 1e-9) * tbytes)
                                           / max(np.sum(tbytes), 1e-9)) if n else float("nan"),
        "mean_transfer_ms": float(np.mean(t_ms)) if n else float("nan"),
        "p50_transfer_ms": float(np.median(t_ms)) if n else float("nan"),
        "exposed_p50_ms": sp[0], "exposed_p90_ms": sp[1], "exposed_p99_ms": sp[2],
        "exposed_bytew_p50_ms": bp[0], "exposed_bytew_p90_ms": bp[1],
        "exposed_bytew_p99_ms": bp[2],
        "mean_hidden_ms": float(np.mean(hidden)) if n else float("nan"),
        "mean_exposed_ms": float(np.mean(exposed)) if n else float("nan"),
    }


def retain_analysis(pairs: pd.DataFrame, models: dict, networks: dict,
                    kv_sweep: tuple[int, list[int]], tp_sweep: list[float],
                    output_dir: Path) -> pd.DataFrame:
    bws = networks["bandwidth_gbps"]
    ohs = networks["fixed_overhead_ms"]
    main_bw = networks["main"]["bandwidth_gbps"]
    main_oh = networks["main"]["fixed_overhead_ms"]

    variants = {
        "unfiltered": pairs,
        "no_suspicious_jump": pairs[~pairs["suspicious_prefix_jump"]],
        "strict_continuity": pairs[pairs["strict_continuity"]],
    }

    # --- Step-level parquet (main overhead, all models x bandwidths, unfiltered).
    step_frames = []
    tokens = pairs["output_tokens"].to_numpy(dtype=np.float64)
    slack = pairs["slack_ms"].to_numpy(dtype=np.float64)
    for mname, m in models.items():
        for bw in bws:
            t_ms = km.transfer_ms(tokens, bytes_per_token=m.logical_kv_bytes_per_token,
                                  block_size_tokens=m.block_size_tokens,
                                  bandwidth_gbps=bw, fixed_overhead_ms=main_oh,
                                  physical_transfer_multiplier=m.physical_transfer_multiplier)
            hidden, exposed, full, frac = km.overlap_split(t_ms, slack)
            step_frames.append(pd.DataFrame({
                "pair_id": pairs["pair_id"].to_numpy(),
                "model": mname, "bandwidth_gbps": bw, "fixed_overhead_ms": main_oh,
                "transfer_ms": t_ms, "hidden_transfer_ms": hidden,
                "exposed_transfer_ms": exposed, "fully_hidden": full,
                "overlap_fraction": frac,
            }))
    step = pd.concat(step_frames, ignore_index=True)
    step.to_parquet(output_dir / "retain_step_results.parquet", index=False)

    # --- Summary: model x bandwidth x variant (+ per-provider / tool-category /
    # trace-model breakdowns at the main network).
    rows = []
    for vname, vdf in variants.items():
        tok = vdf["output_tokens"].to_numpy(dtype=np.float64)
        slk = vdf["slack_ms"].to_numpy(dtype=np.float64)
        for mname, m in models.items():
            for bw in bws:
                rows.append({
                    "group_type": "model_bandwidth", "variant": vname, "model": mname,
                    "bandwidth_gbps": bw, "fixed_overhead_ms": main_oh, "group": "",
                    **_retain_metrics(tok, slk, bpt=m.logical_kv_bytes_per_token,
                                      block=m.block_size_tokens, bw=bw, oh=main_oh,
                                      mult=m.physical_transfer_multiplier),
                })
        for gcol, gtype in [("provider", "provider"), ("tool_category", "tool_category"),
                            ("model", "trace_model")]:
            for gval, gdf in vdf.groupby(gcol):
                if len(gdf) < 30:
                    continue
                for mname, m in models.items():
                    rows.append({
                        "group_type": gtype, "variant": vname, "model": mname,
                        "bandwidth_gbps": main_bw, "fixed_overhead_ms": main_oh,
                        "group": str(gval),
                        **_retain_metrics(gdf["output_tokens"].to_numpy(dtype=np.float64),
                                          gdf["slack_ms"].to_numpy(dtype=np.float64),
                                          bpt=m.logical_kv_bytes_per_token,
                                          block=m.block_size_tokens, bw=main_bw, oh=main_oh,
                                          mult=m.physical_transfer_multiplier),
                    })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary_retain.csv", index=False)

    # --- Sensitivity sweeps (aggregates only, unfiltered variant).
    sens = []
    for mname, m in models.items():
        for bw in bws:
            for oh in ohs:
                sens.append({"sweep": "network", "model": mname, "bandwidth_gbps": bw,
                             "fixed_overhead_ms": oh, "kv_dtype": m.kv_dtype,
                             "transfer_multiplier": m.physical_transfer_multiplier,
                             "scenario": "prefix_resident",
                             **_retain_metrics(tokens, slack,
                                               bpt=m.logical_kv_bytes_per_token,
                                               block=m.block_size_tokens, bw=bw, oh=oh)})
        for dtype, bpt in m.kv_bytes_by_dtype.items():
            sens.append({"sweep": "kv_dtype", "model": mname, "bandwidth_gbps": main_bw,
                         "fixed_overhead_ms": main_oh, "kv_dtype": dtype,
                         "transfer_multiplier": 1.0, "scenario": "prefix_resident",
                         **_retain_metrics(tokens, slack, bpt=bpt,
                                           block=m.block_size_tokens, bw=main_bw, oh=main_oh)})
        for mult in tp_sweep:
            sens.append({"sweep": "tp_replication", "model": mname, "bandwidth_gbps": main_bw,
                         "fixed_overhead_ms": main_oh, "kv_dtype": m.kv_dtype,
                         "transfer_multiplier": mult, "scenario": "prefix_resident",
                         **_retain_metrics(tokens, slack, bpt=m.logical_kv_bytes_per_token,
                                           block=m.block_size_tokens, bw=main_bw,
                                           oh=main_oh, mult=mult)})

        # Prefix-cache-miss scenarios.
        # (a) exclude non-continuity pairs -> covered by strict variant in summary.
        # (b) full prefix KV also travels D->P over the same link.
        full_tokens = (pairs["input_tokens_total"].to_numpy(dtype=np.float64)
                       + tokens)
        for bw in [main_bw, 900]:
            sens.append({"sweep": "prefix_miss", "model": mname, "bandwidth_gbps": bw,
                         "fixed_overhead_ms": main_oh, "kv_dtype": m.kv_dtype,
                         "transfer_multiplier": 1.0,
                         "scenario": "miss_full_prefix_transfer",
                         **_retain_metrics(full_tokens, slack,
                                           bpt=m.logical_kv_bytes_per_token,
                                           block=m.block_size_tokens, bw=bw, oh=main_oh)})
        # (c) restore prefix from a lower-tier cache at restore_bw while the
        # delta travels the network: effective time = max of both (parallel),
        # approximated by the slower restore path dominating.
        for rbw in networks["prefix_restore_bandwidth_gbps"]:
            t_delta = km.transfer_ms(tokens, bytes_per_token=m.logical_kv_bytes_per_token,
                                     block_size_tokens=m.block_size_tokens,
                                     bandwidth_gbps=main_bw, fixed_overhead_ms=main_oh)
            t_restore = km.transfer_ms(
                pairs["input_tokens_total"].to_numpy(dtype=np.float64),
                bytes_per_token=m.logical_kv_bytes_per_token,
                block_size_tokens=m.block_size_tokens,
                bandwidth_gbps=rbw, fixed_overhead_ms=main_oh)
            t_total = np.maximum(t_delta, t_restore)
            hidden, exposed, full, _ = km.overlap_split(t_total, slack)
            sp = _wq(exposed, np.ones(len(exposed)), [0.5, 0.9, 0.99])
            sens.append({"sweep": "prefix_miss", "model": mname, "bandwidth_gbps": rbw,
                         "fixed_overhead_ms": main_oh, "kv_dtype": m.kv_dtype,
                         "transfer_multiplier": 1.0,
                         "scenario": "miss_lower_tier_restore", "n_steps": len(t_total),
                         "fully_hidden_frac": float(np.mean(full)),
                         "mean_transfer_ms": float(np.mean(t_total)),
                         "p50_transfer_ms": float(np.median(t_total)),
                         "exposed_p50_ms": sp[0], "exposed_p90_ms": sp[1],
                         "exposed_p99_ms": sp[2],
                         "mean_hidden_ms": float(np.mean(hidden)),
                         "mean_exposed_ms": float(np.mean(exposed))})

    block, sizes = kv_sweep
    for bpt in sizes:
        for bw in bws:
            sens.append({"sweep": "kv_size_sweep", "model": f"kv{bpt // 1024}KiB",
                         "bandwidth_gbps": bw, "fixed_overhead_ms": main_oh,
                         "kv_dtype": "n/a", "transfer_multiplier": 1.0,
                         "scenario": "prefix_resident",
                         **_retain_metrics(tokens, slack, bpt=bpt, block=block,
                                           bw=bw, oh=main_oh)})
    pd.DataFrame(sens).to_csv(output_dir / "sensitivity_retain.csv", index=False)
    return step


# ---------------------------------------------------------------------------
# 4. Strip-thinking analysis
# ---------------------------------------------------------------------------

def strip_analysis(pairs: pd.DataFrame, models: dict, profiles: dict,
                   prefill_cfg: dict, networks: dict, output_dir: Path,
                   prefill_table: km.PrefillLookupTable | None = None) -> pd.DataFrame:
    main_bw = networks["main"]["bandwidth_gbps"]
    main_oh = networks["main"]["fixed_overhead_ms"]
    main_profile = prefill_cfg["main_profile"]
    main_conc = int(prefill_cfg["main_concurrency"])
    grid = prefill_cfg["benchmark_grid"]

    prefix = pairs["prefix_tokens"].to_numpy(dtype=np.float64)
    # C_i at round-i invocation = full input context of round i.
    ctx = pairs["input_tokens_total"].to_numpy(dtype=np.float64)
    visible = pairs["visible_output_tokens"].to_numpy(dtype=np.float64)
    canon_slack_ms = pairs["canon_slack_ms"].to_numpy(dtype=np.float64)
    est_o = pairs["estimated_tool_append_tokens"].to_numpy(dtype=np.float64)
    est_valid = pairs["est_append_valid"].to_numpy()

    # ---- Phase 1: trace-only required prefill throughput.
    slack_sec = canon_slack_ms / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        required_tps = np.where(visible <= 0, 0.0,
                                np.where(slack_sec > 0, visible / slack_sec, np.inf))

    step = pd.DataFrame({
        "pair_id": pairs["pair_id"].to_numpy(),
        "provider": pairs["provider"].to_numpy(),
        "visible_kind": np.where(pairs["visible_is_upper_bound"], "upper_bound", "exact"),
        "canonicalization_slack_ms": canon_slack_ms,
        "required_prefill_tokens_per_sec": required_tps,
    })

    # ---- Phase 2: S0/S1/S2 under synthetic profiles (or a measured table).
    def make_xfer(m, bw, oh):
        return lambda tok: km.transfer_ms(
            tok, bytes_per_token=m.logical_kv_bytes_per_token,
            block_size_tokens=m.block_size_tokens, bandwidth_gbps=bw,
            fixed_overhead_ms=oh, physical_transfer_multiplier=m.physical_transfer_multiplier)

    o_tokens = np.where(est_valid, est_o, np.nan)
    step_model_frames = []
    sens_rows = []
    for mname, m in models.items():
        for pname, prof in profiles.items():
            if prefill_table is not None:
                prefill = prefill_table
                pname = "measured_table"
            else:
                prefill = km.table_from_profile(
                    m, prof, prefix_grid=grid["prefix_tokens"],
                    suffix_grid=grid["suffix_tokens"], concurrency=main_conc)
            res = km.strip_scenarios(
                prefix_tokens=ctx, visible_tokens=visible,
                tool_append_tokens=np.nan_to_num(o_tokens, nan=0.0),
                canon_slack_ms=canon_slack_ms, prefill=prefill,
                xfer=make_xfer(m, main_bw, main_oh))
            shadow_hidden = res["exposed_shadow_ms"] <= 0
            if pname == main_profile or prefill_table is not None:
                step_model_frames.append(pd.DataFrame({
                    "pair_id": pairs["pair_id"].to_numpy(),
                    "model": mname, "profile": pname,
                    "shadow_prefill_ms": res["shadow_prefill_ms"],
                    "exposed_shadow_ms": res["exposed_shadow_ms"],
                    "shadow_fully_hidden": shadow_hidden,
                    "T_S0_ms": np.where(est_valid, res["T_S0"], np.nan),
                    "T_S1_ms": np.where(est_valid, res["T_S1"], np.nan),
                    "T_S2_ms": np.where(est_valid, res["T_S2"], np.nan),
                    "saving_S1_ms": np.where(est_valid, res["saving_S1"], np.nan),
                    "saving_S2_ms": np.where(est_valid, res["saving_S2"], np.nan),
                }))
            # Queue sensitivity (aggregates).
            for q in networks["p_queue_ms"]:
                resq = km.strip_scenarios(
                    prefix_tokens=ctx, visible_tokens=visible,
                    tool_append_tokens=np.nan_to_num(o_tokens, nan=0.0),
                    canon_slack_ms=canon_slack_ms, prefill=prefill,
                    xfer=make_xfer(m, main_bw, main_oh),
                    q_reactive_ms=q, q_proactive_ms=q)
                valid = est_valid
                sens_rows.append({
                    "model": mname, "profile": pname, "p_queue_ms": q,
                    "concurrency": main_conc, "n_steps": int(valid.sum()),
                    "shadow_fully_hidden_frac": float(np.mean(resq["exposed_shadow_ms"] <= 0)),
                    "saving_S1_p50_ms": float(np.nanmedian(np.where(valid, resq["saving_S1"], np.nan))),
                    "saving_S2_p50_ms": float(np.nanmedian(np.where(valid, resq["saving_S2"], np.nan))),
                    "s2_worse_than_s1_frac": float(np.nanmean(
                        np.where(valid, resq["T_S2"] > resq["T_S1"], np.nan))),
                    "s2_worse_than_s0_frac": float(np.nanmean(
                        np.where(valid, resq["T_S2"] > resq["T_S0"], np.nan))),
                })

    phase2 = pd.concat(step_model_frames, ignore_index=True)
    step = step.merge(phase2, on="pair_id", how="right")
    step.to_parquet(output_dir / "strip_step_results.parquet", index=False)
    pd.DataFrame(sens_rows).to_csv(output_dir / "sensitivity_prefill_queue.csv", index=False)

    # ---- Summary CSV.
    rows = []
    qs = [0.5, 0.9, 0.99]
    meta = pairs[["pair_id", "model", "tool_category", "prefix_tokens",
                  "visible_output_tokens", "canon_slack_ms", "est_append_valid",
                  "visible_is_upper_bound"]].rename(columns={"model": "trace_model"})
    merged = step.merge(meta, on="pair_id")

    def bin_prefix(v):
        edges = [0, 16_384, 65_536, 131_072, 262_144, 1 << 60]
        labels = ["<16K", "16-64K", "64-128K", "128-256K", ">=256K"]
        return pd.cut(v, bins=edges, labels=labels, right=False)

    def bin_visible(v):
        edges = [0, 64, 256, 1024, 4096, 1 << 60]
        labels = ["<64", "64-256", "256-1K", "1-4K", ">=4K"]
        return pd.cut(v, bins=edges, labels=labels, right=False)

    merged["prefix_bin"] = bin_prefix(merged["prefix_tokens"])
    merged["visible_bin"] = bin_visible(merged["visible_output_tokens"])

    group_specs = [
        ("provider", "provider"), ("trace_model", "trace_model"),
        ("tool_category", "tool_category"), ("prefix_bin", "prefix_bin"),
        ("visible_bin", "visible_bin"),
    ]
    for mname in merged["model"].unique():
        mdf = merged[merged["model"] == mname]
        for gtype, gcol in [("all", None)] + group_specs:
            groups = [("all", mdf)] if gcol is None else list(mdf.groupby(gcol, observed=True))
            for gval, gdf in groups:
                if len(gdf) < 30:
                    continue
                req = gdf["required_prefill_tokens_per_sec"].to_numpy()
                finite = req[np.isfinite(req)]
                inf_frac = float(np.mean(~np.isfinite(req))) if len(req) else float("nan")
                sv1 = gdf["saving_S1_ms"].dropna()
                sv2 = gdf["saving_S2_ms"].dropna()
                rows.append({
                    "model": mname, "group_type": gtype, "group": str(gval),
                    "n_steps": len(gdf),
                    "exact_frac": float(np.mean(~gdf["visible_is_upper_bound"])),
                    "required_tps_p50": float(np.quantile(finite, 0.5)) if len(finite) else np.nan,
                    "required_tps_p90": float(np.quantile(finite, 0.9)) if len(finite) else np.nan,
                    "required_tps_p99": float(np.quantile(finite, 0.99)) if len(finite) else np.nan,
                    "zero_slack_frac": inf_frac,
                    "shadow_fully_hidden_frac": float(gdf["shadow_fully_hidden"].mean()),
                    "exposed_shadow_p50_ms": float(gdf["exposed_shadow_ms"].quantile(0.5)),
                    "exposed_shadow_p90_ms": float(gdf["exposed_shadow_ms"].quantile(0.9)),
                    "exposed_shadow_p99_ms": float(gdf["exposed_shadow_ms"].quantile(0.99)),
                    "n_e2e_valid": int(sv1.notna().sum()),
                    "T_S0_p50_ms": float(gdf["T_S0_ms"].quantile(0.5)),
                    "T_S1_p50_ms": float(gdf["T_S1_ms"].quantile(0.5)),
                    "T_S2_p50_ms": float(gdf["T_S2_ms"].quantile(0.5)),
                    "saving_S1_p50_ms": float(sv1.quantile(0.5)) if len(sv1) else np.nan,
                    "saving_S1_p90_ms": float(sv1.quantile(0.9)) if len(sv1) else np.nan,
                    "saving_S2_p50_ms": float(sv2.quantile(0.5)) if len(sv2) else np.nan,
                    "saving_S2_p90_ms": float(sv2.quantile(0.9)) if len(sv2) else np.nan,
                    "s2_worse_than_s1_frac": float(
                        (gdf["T_S2_ms"] > gdf["T_S1_ms"]).mean()) if len(sv2) else np.nan,
                })
    pd.DataFrame(rows).to_csv(output_dir / "summary_strip.csv", index=False)
    return step


# ---------------------------------------------------------------------------
# 5. Request / session level summaries
# ---------------------------------------------------------------------------

def request_session_summaries(pairs: pd.DataFrame, retain_step: pd.DataFrame,
                              strip_step: pd.DataFrame, models: dict,
                              networks: dict, main_profile: str,
                              output_dir: Path) -> None:
    main_bw = networks["main"]["bandwidth_gbps"]
    ret = retain_step[(retain_step["bandwidth_gbps"] == main_bw)]
    ret = ret.pivot_table(index="pair_id", columns="model",
                          values=["hidden_transfer_ms", "transfer_ms"], aggfunc="first")
    ret.columns = [f"retain_{a}_{b}" for a, b in ret.columns]

    strip = strip_step.pivot_table(index="pair_id", columns="model",
                                   values=["saving_S1_ms", "saving_S2_ms"], aggfunc="first")
    strip.columns = [f"strip_{a}_{b}" for a, b in strip.columns]

    req = pairs[["pair_id", "provider", "project", "session_id", "session_file",
                 "round_index", "model", "response_time_ms", "slack_ms",
                 "canon_slack_ms"]].set_index("pair_id")
    req = req.join(ret).join(strip)
    for mname in models:
        hid = req[f"retain_hidden_transfer_ms_{mname}"]
        xfer = req[f"retain_transfer_ms_{mname}"]
        # Reactive baseline puts the whole transfer on the response path.
        req[f"retain_saving_frac_{mname}"] = hid / (req["response_time_ms"] + xfer)
        sv1 = req.get(f"strip_saving_S1_ms_{mname}")
        if sv1 is not None:
            req[f"strip_saving_S1_frac_{mname}"] = sv1.clip(lower=0) / np.maximum(
                req["response_time_ms"] + sv1.clip(lower=0), 1e-9)
    req.reset_index().to_csv(output_dir / "request_level_summary.csv", index=False)

    session_key = ["provider", "project", "session_id", "session_file"]
    agg = {"response_time_ms": "sum", "slack_ms": "sum"}
    for mname in models:
        agg[f"retain_hidden_transfer_ms_{mname}"] = "sum"
        agg[f"retain_transfer_ms_{mname}"] = "sum"
        col = f"strip_saving_S1_ms_{mname}"
        if col in req.columns:
            agg[col] = "sum"
        col2 = f"strip_saving_S2_ms_{mname}"
        if col2 in req.columns:
            agg[col2] = "sum"
    ses = req.reset_index().groupby(session_key, dropna=False).agg(
        n_pairs=("pair_id", "count"), **{k: (k, v) for k, v in agg.items()})
    for mname in models:
        ses[f"retain_saving_frac_{mname}"] = (
            ses[f"retain_hidden_transfer_ms_{mname}"]
            / (ses["response_time_ms"] + ses[f"retain_transfer_ms_{mname}"]))
    ses.reset_index().to_csv(output_dir / "session_level_summary.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Validation + audit
# ---------------------------------------------------------------------------

def validate(con, rounds: pd.DataFrame, pairs: pd.DataFrame, output_dir: Path) -> None:
    checks = []

    def add(name, computed, expected, tol_frac=0.15):
        ok = (np.isfinite(computed) and expected != 0
              and abs(computed - expected) / abs(expected) <= tol_frac)
        checks.append({"check": name, "computed": computed, "expected": expected,
                       "ok": bool(ok)})

    n_rounds = con.execute("SELECT count(*) FROM rounds").fetchone()[0]
    n_tools = con.execute("SELECT count(*) FROM tool_calls").fetchone()[0]
    add("readme_round_count", n_rounds, README_ROUND_COUNT, 0.0)
    add("readme_tool_count", n_tools, README_TOOL_COUNT, 0.0)

    for (prov, kind), expected in PAPER_TABLE8_MEDIANS.items():
        col = {"prefix": "prefix_tokens", "append": "newly_append_tokens",
               "output": "output_tokens"}[kind]
        computed = float(rounds.loc[rounds["provider"] == prov, col].median())
        add(f"table8_{prov}_{kind}_median", computed, expected, 0.15)

    # Table 7: per-step tool execution wall span; per-request = sum over the
    # rounds of one user-triggered request. Per-step here uses positive spans.
    span_s = (rounds["tools_ready_us"] - rounds["first_emit_us"]) / 1e6
    span_s = span_s[(rounds["n_tools"] > 0) & span_s.notna() & (span_s > 0)]
    add("table7_per_step_tool_p50_s", float(span_s.quantile(0.5)),
        PAPER_TABLE7_TOOL_EXEC["per_step_p50_s"], 3.0)
    add("table7_per_step_tool_p90_s", float(span_s.quantile(0.9)),
        PAPER_TABLE7_TOOL_EXEC["per_step_p90_s"], 3.0)

    # tool_wall_latency_ms == result_at - emitted_at (sampled).
    sample = con.execute("""
        SELECT tool_wall_latency_ms,
               CAST(epoch_us(result_at) - epoch_us(emitted_at) AS BIGINT) / 1000 AS diff_ms
        FROM tool_calls WHERE result_at IS NOT NULL AND emitted_at IS NOT NULL
          AND tool_wall_latency_ms IS NOT NULL
        USING SAMPLE reservoir(2000 ROWS) REPEATABLE (42)
    """).df()
    mismatch = float(np.mean(np.abs(sample["tool_wall_latency_ms"] - sample["diff_ms"]) > 2))
    checks.append({"check": "tool_wall_latency_matches_result_minus_emitted",
                   "computed": mismatch, "expected": 0.0, "ok": mismatch < 0.01})

    # Output-cached vs output-resend trend vs paper (Claude resend, gpt-5.4
    # cached, gpt-5.5 resend) on pairs with meaningful output.
    big = pairs[pairs["output_tokens"] >= 2000]
    for sel, name, expect in [
        (big["provider"] == "claude", "claude", "output_resend_like"),
        ((big["provider"] == "codex") & (big["model"] == "gpt-5.4"), "gpt-5.4",
         "output_cached_like"),
        ((big["provider"] == "codex") & (big["model"] == "gpt-5.5"), "gpt-5.5",
         "output_resend_like"),
    ]:
        sub = big[sel]
        if len(sub) < 50:
            continue
        top = sub["output_assignment"].value_counts(normalize=True)
        checks.append({"check": f"assignment_trend_{name}",
                       "computed": f"{top.index[0]}={top.iloc[0]:.2f}",
                       "expected": expect, "ok": top.index[0] == expect})

    dfc = pd.DataFrame(checks)
    dfc.to_csv(output_dir / "validation_report.csv", index=False)
    print(dfc.to_string(index=False), file=sys.stderr)


def audit_sample(pairs: pd.DataFrame, output_dir: Path, n: int = 50) -> None:
    sample = pairs.sample(n=min(n, len(pairs)), random_state=42)
    cols = ["pair_id", "provider", "project", "session_id", "round_index", "model",
            "round_pk", "next_round_pk", "n_tools", "tool_names", "critical_tool_name",
            "tool_category", "parallel_tools", "min_tool_eff_ms", "max_tool_eff_ms",
            "sum_tool_eff_ms", "tool_wall_span_ms", "slack_ms", "canon_slack_ms",
            "tool_ready_source_diff_ms", "output_tokens", "reasoning_output_tokens",
            "visible_output_tokens", "prefix_tokens", "input_tokens_total",
            "next_prefix_tokens", "next_newly_append_tokens", "prefix_delta",
            "output_assignment", "estimated_tool_append_tokens",
            "suspicious_prefix_jump", "strict_continuity", "response_time_ms"]
    records = json.loads(sample[cols].to_json(orient="records"))
    with (output_dir / "audit_sample_50.json").open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    trace_db.add_db_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs-dir", type=Path, default=EXP_DIR / "configs")
    parser.add_argument("--prefill-table", type=Path, default=None,
                        help="measured prefill lookup CSV "
                             "(prefix_tokens,suffix_tokens,prefill_ms)")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = km.load_models(args.configs_dir)
    kv_sweep = km.load_kv_size_sweep(args.configs_dir)
    networks = km.load_networks(args.configs_dir)
    prefill_cfg = km.load_prefill_profiles(args.configs_dir)
    profiles = km.parse_profiles(prefill_cfg)
    tp_sweep = km.load_yaml(args.configs_dir / "models.yaml")["tp_replication_sweep"]

    prefill_table = None
    if args.prefill_table is not None:
        tbl = pd.read_csv(args.prefill_table)
        piv = tbl.pivot_table(index="prefix_tokens", columns="suffix_tokens",
                              values="prefill_ms")
        prefill_table = km.PrefillLookupTable(
            list(piv.index), list(piv.columns), piv.to_numpy())

    con = trace_db.open_from_args(args)
    print("Extracting round features...", file=sys.stderr)
    rounds = extract_round_features(con)
    print(f"  {len(rounds):,} rounds", file=sys.stderr)

    print("Building pairs...", file=sys.stderr)
    pairs, excluded, stages = build_pairs(rounds)
    print(f"  eligible={len(pairs):,} excluded={len(excluded):,}", file=sys.stderr)
    print(stages.to_string(index=False), file=sys.stderr)

    pairs_out = pairs.drop(columns=[c for c in pairs.columns
                                    if c.startswith("next_") and c not in
                                    {"next_round_pk", "next_prefix_tokens",
                                     "next_newly_append_tokens", "next_input_tokens_total",
                                     "next_round_index"}])
    pairs_out.to_parquet(output_dir / "eligible_pairs.parquet", index=False)
    excluded.to_csv(output_dir / "excluded_pairs.csv", index=False)
    stages.to_csv(output_dir / "filter_stages.csv", index=False)

    print("Retain-thinking analysis...", file=sys.stderr)
    retain_step = retain_analysis(pairs, models, networks, kv_sweep, tp_sweep, output_dir)

    print("Strip-thinking analysis...", file=sys.stderr)
    strip_step = strip_analysis(pairs, models, profiles, prefill_cfg, networks,
                                output_dir, prefill_table)

    print("Request/session summaries...", file=sys.stderr)
    request_session_summaries(pairs, retain_step, strip_step, models, networks,
                              prefill_cfg["main_profile"], output_dir)

    audit_sample(pairs, output_dir)
    if not args.skip_validation:
        print("Validation...", file=sys.stderr)
        validate(con, rounds, pairs, output_dir)

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
