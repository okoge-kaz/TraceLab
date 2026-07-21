#!/usr/bin/env python3
"""Workload phase decomposition from measured timestamps: prefill(TTFT) /
decode / tool execution per step-cycle, plus simulated KV-transfer width,
rendered as (a) a real-session Gantt and (b) composition bars.

Run:
    uv run --with pandas --with pyarrow --with pyyaml python \
        artifacts/kv_overlap/plot_timeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "artifacts" / "utils"))
sys.path.insert(0, str(EXP_DIR))

import png_sidecar  # noqa: E402
from style import plt, save_plot, polish_axes, PLOT_COLORS, MUTED_TEXT, TEXT_COLOR  # noqa: E402

import kv_models as km  # noqa: E402

PHASE_COLORS = {
    "prefill_ttft": "#2563eb",   # queue + network + prefill (TTFT proxy)
    "decode": "#059669",
    "kv_transfer": "#dc2626",
    "tool": "#d97706",
}
PHASE_LABELS = {
    "prefill_ttft": "Input→1st output event (queue+prefill+1st block decode)",
    "decode": "1st→last output event (remaining decode)",
    "kv_transfer": "KV transfer (simulated, Qwen3 @100GB/s)",
    "tool": "Tool execution (slack)",
}


def load_cycles(outdir: Path) -> pd.DataFrame:
    p = pd.read_parquet(outdir / "eligible_pairs.parquet")
    # Tool-result-initiated rounds only: TTFT measurable from own input events.
    df = p[p["first_input_event_type"] == "tool_result"].copy()
    df["ttft_ms"] = (df["gen_start_us"] - df["input_tool_result_last_us"]) / 1000.0
    df["decode_ms"] = (df["gen_end_us"] - df["gen_start_us"]) / 1000.0
    df["tool_ms"] = df["slack_ms"]
    df = df[(df["ttft_ms"] >= 0) & (df["decode_ms"] >= 0)]
    m = km.load_models()["qwen3-235b-a22b"]
    df["kv_ms"] = km.transfer_ms(
        df["output_tokens"].to_numpy(dtype=float),
        bytes_per_token=m.logical_kv_bytes_per_token,
        block_size_tokens=m.block_size_tokens,
        bandwidth_gbps=100.0, fixed_overhead_ms=0.5)
    df["cycle_ms"] = df["ttft_ms"] + df["decode_ms"] + df["tool_ms"]
    return df


def plot_composition(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    phases = ["prefill_ttft", "decode", "kv_transfer", "tool"]
    cols = {"prefill_ttft": "ttft_ms", "decode": "decode_ms",
            "kv_transfer": "kv_ms", "tool": "tool_ms"}

    # Left: per-step medians (the "typical step" — medians don't sum to the
    # median cycle, but show typical magnitude per phase).
    ax = axes[0]
    rows = []
    for prov, g in df.groupby("provider"):
        rows.append((prov, [g[cols[ph]].median() for ph in phases]))
    ypos = np.arange(len(rows))
    for j, ph in enumerate(phases):
        vals = [r[1][j] for r in rows]
        ax.barh(ypos + (j - 1.5) * 0.2, vals, height=0.18,
                color=PHASE_COLORS[ph], label=PHASE_LABELS[ph])
    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("median duration per step (ms, log scale)")
    ax.set_title("Typical (median) phase durations")
    for j, ph in enumerate(phases):
        for i, r in enumerate(rows):
            v = r[1][j]
            ax.text(v * 1.15, i + (j - 1.5) * 0.2,
                    f"{v:,.1f} ms" if v < 10_000 else f"{v/1000:,.1f} s",
                    va="center", fontsize=7.5, color=TEXT_COLOR)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.06))
    polish_axes(ax, grid_axis="x")

    # Right: share of total wall-clock (sums, per provider) — "where does the
    # step-cycle time actually go in aggregate".
    ax = axes[1]
    for i, (prov, g) in enumerate(df.groupby("provider")):
        total = g[["ttft_ms", "decode_ms", "tool_ms"]].sum().sum() + g["kv_ms"].sum()
        left = 0.0
        for ph in phases:
            share = g[cols[ph]].sum() / total * 100
            ax.barh(i, share, left=left, color=PHASE_COLORS[ph])
            if share > 4:
                ax.text(left + share / 2, i, f"{share:.1f}%", ha="center",
                        va="center", fontsize=9, color="white", fontweight="bold")
            elif share > 0.05:
                ax.text(left + share + 1, i + 0.28, f"{PHASE_LABELS[ph].split(' (')[0]}: {share:.2f}%",
                        fontsize=7, color=PHASE_COLORS[ph])
            left += share
    ax.set_yticks(range(df["provider"].nunique()),
                  sorted(df["provider"].unique()))
    ax.set_xlabel("share of total step-cycle wall-clock (%)")
    ax.set_title("Aggregate time share (sum over all steps)")
    ax.set_xlim(0, 100)
    polish_axes(ax, grid_axis="x")

    fig.suptitle("Measured step-cycle decomposition — tool-initiated steps "
                 f"(n={len(df):,}; KV transfer simulated)", fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_plot(fig, outdir / "workload_phase_composition.png")


def pick_gantt_session(df: pd.DataFrame, *, min_rounds=5, max_cycle_s=120.0):
    """A real session segment: consecutive eligible tool cycles, readable span."""
    key = ["provider", "project", "session_id", "session_file"]
    best = None
    for (prov, *_), g in df.groupby(key, sort=False):
        if len(g) < min_rounds:
            continue
        g = g.sort_values("round_index")
        # find a run of consecutive round_index
        idx = g["round_index"].to_numpy()
        run_start = 0
        for i in range(1, len(idx) + 1):
            if i == len(idx) or idx[i] != idx[i - 1] + 1:
                if i - run_start >= min_rounds:
                    seg = g.iloc[run_start:i].head(8)
                    span = (seg["tool_ready_us"].max() - seg["input_tool_result_last_us"].min()) / 1e6
                    med_tool = seg["tool_ms"].median()
                    if 5.0 < span < max_cycle_s and med_tool > 400:
                        score = abs(span - 30.0) + (0 if prov == "codex" else 20)
                        if best is None or score < best[0]:
                            best = (score, seg)
                run_start = i
    return None if best is None else best[1]


def plot_gantt(df: pd.DataFrame, outdir: Path) -> None:
    seg = pick_gantt_session(df)
    if seg is None:
        print("no suitable session found for gantt", file=sys.stderr)
        return
    t0 = seg["input_tool_result_last_us"].min()

    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    lanes = {"prefill_ttft": 3, "decode": 2, "kv_transfer": 1, "tool": 0}

    for _, r in seg.iterrows():
        in_s = (r["input_tool_result_last_us"] - t0) / 1e6
        gs = (r["gen_start_us"] - t0) / 1e6
        ge = (r["gen_end_us"] - t0) / 1e6
        tr = (r["tool_ready_us"] - t0) / 1e6
        kv_s = r["kv_ms"] / 1000.0

        ax.barh(lanes["prefill_ttft"], gs - in_s, left=in_s, height=0.6,
                color=PHASE_COLORS["prefill_ttft"], alpha=0.9)
        ax.barh(lanes["decode"], ge - gs, left=gs, height=0.6,
                color=PHASE_COLORS["decode"], alpha=0.9)
        # KV transfer starts right at generation end (proactive); simulated.
        ax.barh(lanes["kv_transfer"], max(kv_s, 0.05), left=ge, height=0.6,
                color=PHASE_COLORS["kv_transfer"], alpha=0.9)
        ax.barh(lanes["tool"], max(tr - ge, 0.02), left=ge, height=0.6,
                color=PHASE_COLORS["tool"], alpha=0.9)
        ax.plot([tr, tr], [-0.5, 3.5], color=MUTED_TEXT, linewidth=0.6,
                linestyle=":", alpha=0.7)
        label = str(r["critical_tool_name"] or "")
        if label:
            ax.text(ge + max(tr - ge, 0.02) / 2, lanes["tool"] - 0.55, label,
                    ha="center", fontsize=6.5, color=MUTED_TEXT)

    ax.set_yticks([3, 2, 1, 0],
                  ["Prefill+1st block\n(input→1st output)", "Decode\n(1st→last output)",
                   "Network (KV)\n[simulated,\nbar ≥50ms floor]", "Tool"])
    ax.set_xlabel("session time (s)")
    kv_total = seg["kv_ms"].sum()
    ax.set_title(
        f"Real {seg['provider'].iloc[0]} session segment ({len(seg)} consecutive tool-initiated steps, "
        f"model={seg['model'].iloc[0]}) — measured timestamps; KV lane simulated "
        f"(Qwen3 @100 GB/s, actual total {kv_total:.1f} ms ≈ invisible at this scale)",
        fontsize=10)
    polish_axes(ax, grid_axis="x")
    fig.tight_layout()
    save_plot(fig, outdir / "workload_phase_gantt.png")

    cols = ["round_index", "ttft_ms", "decode_ms", "tool_ms", "kv_ms",
            "output_tokens", "prefix_tokens", "n_tools", "critical_tool_name"]
    seg[cols].to_csv(outdir / "workload_phase_gantt_sample.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=EXP_DIR / "outputs")
    args = parser.parse_args()
    df = load_cycles(args.output_dir)
    print(f"{len(df):,} tool-initiated cycles", file=sys.stderr)
    for prov, g in df.groupby("provider"):
        tot = g[["ttft_ms", "decode_ms", "tool_ms"]].sum().sum() + g["kv_ms"].sum()
        print(f"{prov}: median ttft={g.ttft_ms.median():.0f}ms decode={g.decode_ms.median():.0f}ms "
              f"tool={g.tool_ms.median():.0f}ms kv={g.kv_ms.median():.2f}ms | "
              f"share ttft={g.ttft_ms.sum()/tot*100:.1f}% decode={g.decode_ms.sum()/tot*100:.1f}% "
              f"tool={g.tool_ms.sum()/tot*100:.1f}% kv={g.kv_ms.sum()/tot*100:.3f}%",
              file=sys.stderr)
    plot_composition(df, args.output_dir)
    plot_gantt(df, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
