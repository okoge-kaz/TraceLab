"""Unit tests for pair extraction timing semantics (parallel tools, slack,
exclusion cascade) on synthetic rounds."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze


def _round(pk, idx, *, n_tools=0, tools_ready_us=None, first_emit_us=None,
           gen_start_us=1_000_000, gen_end_us=2_000_000, a_ready_us=None,
           sum_tool_eff_ms=None, output_tokens=100, reasoning=20,
           input_total=10_000, prefix=9_000, append=1_000,
           first_input="tool_result", user_msgs=0, next_tool_result_us=None,
           provider="codex", session="s1"):
    return {
        "round_pk": pk, "ingest_seq": pk, "provider": provider, "project": "p",
        "session_id": session, "session_file": "f", "round_index": idx,
        "round_id": f"r{pk}", "model": "gpt-5.5", "turn_id": None,
        "input_tokens_total": input_total, "prefix_tokens": prefix,
        "newly_append_tokens": append, "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning, "current_user_message_count": user_msgs,
        "current_tool_result_count": 1 if first_input == "tool_result" else 0,
        "first_input_event_type": first_input,
        "gen_end_us": gen_end_us, "gen_start_us": gen_start_us,
        "a_ready_us": a_ready_us if a_ready_us is not None else gen_end_us,
        "first_ev_us": gen_start_us - 100_000, "last_ev_us": gen_end_us,
        "user_msg_events": user_msgs,
        "input_tool_result_events": 1 if first_input == "tool_result" else 0,
        "input_tool_result_last_us": next_tool_result_us,
        "n_tools": n_tools, "n_tools_missing_result": 0,
        "tools_ready_us": tools_ready_us, "first_emit_us": first_emit_us,
        "last_emit_us": first_emit_us, "min_tool_eff_ms": 10.0,
        "max_tool_eff_ms": 10.0, "sum_tool_eff_ms": sum_tool_eff_ms,
        "critical_tool_name": "exec_command" if n_tools else None,
        "any_tool_error": False, "tool_names": "exec_command",
        "tool_category": "exec" if n_tools else "other",
    }


def _pairs(rows):
    eligible, excluded, stages = analyze.build_pairs(pd.DataFrame(rows))
    return eligible, excluded


def test_parallel_tools_use_max_completion_not_sum():
    # Two parallel tools: each 5s, run concurrently -> tools_ready is max
    # result_at (7s), NOT emitted+sum (12s). We feed tools_ready = max already
    # (SQL max(result_at)); the test asserts slack uses it and flags parallel.
    rows = [
        _round(1, 0, n_tools=2, tools_ready_us=7_000_000, first_emit_us=2_000_000,
               gen_end_us=2_000_000, sum_tool_eff_ms=10_000.0,
               next_tool_result_us=None),
        _round(2, 1, input_total=11_000, prefix=10_100, append=900,
               next_tool_result_us=7_000_000, gen_start_us=7_100_000,
               gen_end_us=8_000_000),
    ]
    eligible, _ = _pairs(rows)
    assert len(eligible) == 1
    row = eligible.iloc[0]
    assert row["tool_ready_us"] == 7_000_000
    assert abs(row["slack_ms"] - 5000.0) < 1e-6      # 7s - 2s, not 12s - 2s
    assert bool(row["parallel_tools"])                # sum(10s) > wall span(5s)


def test_conservative_tool_ready_uses_later_source():
    # result_at says 5s but the next round's tool_result event says 6s ->
    # conservative ready time is 6s.
    rows = [
        _round(1, 0, n_tools=1, tools_ready_us=5_000_000, first_emit_us=2_000_000),
        _round(2, 1, input_total=11_000, prefix=10_100, append=900,
               next_tool_result_us=6_000_000, gen_start_us=6_100_000,
               gen_end_us=7_000_000),
    ]
    eligible, _ = _pairs(rows)
    row = eligible.iloc[0]
    assert row["tool_ready_us"] == 6_000_000
    assert abs(row["tool_ready_source_diff_ms"] - 1000.0) < 1e-6
    assert abs(row["slack_ms"] - 4000.0) < 1e-6


def test_negative_slack_clipped_to_zero():
    # Tool completes before the last output event -> slack 0, not negative.
    rows = [
        _round(1, 0, n_tools=1, tools_ready_us=1_500_000, first_emit_us=1_200_000,
               gen_end_us=2_000_000),
        _round(2, 1, input_total=11_000, prefix=10_100, append=900,
               gen_start_us=2_100_000, gen_end_us=3_000_000),
    ]
    eligible, _ = _pairs(rows)
    assert eligible.iloc[0]["slack_ms"] == 0.0


def test_canonicalization_slack_uses_last_non_reasoning_event():
    # a_ready (last text/tool_call) at 1.5s, gen_end (incl. trailing reasoning)
    # at 2s, tools ready at 5s -> canon slack 3.5s > slack 3s.
    rows = [
        _round(1, 0, n_tools=1, tools_ready_us=5_000_000, first_emit_us=1_500_000,
               gen_end_us=2_000_000, a_ready_us=1_500_000),
        _round(2, 1, input_total=11_000, prefix=10_100, append=900,
               gen_start_us=5_100_000, gen_end_us=6_000_000),
    ]
    eligible, _ = _pairs(rows)
    row = eligible.iloc[0]
    assert abs(row["slack_ms"] - 3000.0) < 1e-6
    assert abs(row["canon_slack_ms"] - 3500.0) < 1e-6


def test_exclusion_reasons():
    rows = [
        # no_tool round followed by another round
        _round(1, 0, n_tools=0),
        # tool round whose next round is user-initiated -> user_intervened
        _round(2, 1, n_tools=1, tools_ready_us=5_000_000, first_emit_us=2_000_000),
        _round(3, 2, first_input="user_message", user_msgs=1,
               gen_start_us=9_000_000, gen_end_us=9_500_000,
               input_total=11_000, prefix=10_100, append=900),
        # last round with tools -> missing_next_round
        _round(4, 3, n_tools=1, tools_ready_us=11_000_000, first_emit_us=10_000_000,
               gen_start_us=9_600_000, gen_end_us=9_900_000),
    ]
    _, excluded = _pairs(rows)
    reasons = dict(zip(excluded["round_pk"], excluded["exclusion_reason"]))
    assert reasons[1] == "no_tool"
    assert reasons[2] == "user_intervened"
    assert reasons[3] == "no_tool"               # mid-session, no tools
    assert reasons[4] == "missing_next_round"    # terminal, had tools


def test_compaction_and_context_reduction():
    rows = [
        _round(1, 0, n_tools=1, tools_ready_us=5_000_000, first_emit_us=2_000_000,
               input_total=100_000, prefix=90_000, append=10_000),
        # next round context collapsed to 30% -> compaction
        _round(2, 1, input_total=30_000, prefix=20_000, append=10_000,
               gen_start_us=5_100_000, gen_end_us=6_000_000, n_tools=1,
               tools_ready_us=8_000_000, first_emit_us=6_000_000),
        # next round mildly shrunk (-5k) -> context_reduction
        _round(3, 2, input_total=25_000, prefix=20_000, append=5_000,
               gen_start_us=8_100_000, gen_end_us=9_000_000),
    ]
    _, excluded = _pairs(rows)
    reasons = dict(zip(excluded["round_pk"], excluded["exclusion_reason"]))
    assert reasons[1] == "compaction"
    assert reasons[2] == "context_reduction"


def test_codex_visible_and_upper_bound_flag():
    rows = [
        _round(1, 0, n_tools=1, tools_ready_us=5_000_000, first_emit_us=2_000_000,
               output_tokens=100, reasoning=60),
        _round(2, 1, input_total=11_000, prefix=10_100, append=900,
               gen_start_us=5_100_000, gen_end_us=6_000_000),
    ]
    eligible, _ = _pairs(rows)
    row = eligible.iloc[0]
    assert row["visible_output_tokens"] == 40.0
    assert not row["visible_is_upper_bound"]

    rows = [
        _round(1, 0, n_tools=1, tools_ready_us=5_000_000, first_emit_us=2_000_000,
               output_tokens=100, reasoning=None, provider="claude"),
        _round(2, 1, input_total=11_000, prefix=10_100, append=900,
               gen_start_us=5_100_000, gen_end_us=6_000_000, provider="claude",
               reasoning=None),
    ]
    eligible, _ = _pairs(rows)
    row = eligible.iloc[0]
    assert row["visible_output_tokens"] == 100.0  # upper bound = full output
    assert row["visible_is_upper_bound"]


def test_suspicious_prefix_jump_flagged():
    rows = [
        _round(1, 0, n_tools=1, tools_ready_us=5_000_000, first_emit_us=2_000_000,
               input_total=10_000, prefix=9_000, output_tokens=100),
        # next prefix jumped to 60k >> prev_total + output -> suspicious
        _round(2, 1, input_total=61_000, prefix=60_000, append=1_000,
               gen_start_us=5_100_000, gen_end_us=6_000_000),
    ]
    eligible, _ = _pairs(rows)
    row = eligible.iloc[0]
    assert bool(row["suspicious_prefix_jump"])
    assert not bool(row["strict_continuity"])
