#!/usr/bin/env python3
"""Plots for the KV-overlap replay analysis (reads analyze.py outputs).

Run:
    uv run --with pandas --with pyarrow --with pyyaml python \
        artifacts/kv_overlap/plot.py
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
from style import plt, save_plot, polish_axes, plot_color, PLOT_COLORS, MUTED_TEXT  # noqa: E402

import kv_models as km  # noqa: E402

MODEL_ORDER = ["qwen3-235b-a22b", "glm-5.2", "kimi-k3-proxy-k2"]
MODEL_LABELS = {
    "qwen3-235b-a22b": "Qwen3-235B-A22B (GQA, 188 KiB/tok)",
    "glm-5.2": "GLM-5.2 (MLA, 88 KiB/tok)",
    "kimi-k3-proxy-k2": "Kimi-K3 proxy=K2 (MLA, 69 KiB/tok)",
}


def _cdf(ax, values, label, color, weights=None, **kw):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    if len(values) == 0:
        return
    order = np.argsort(values)
    v = values[order]
    if weights is None:
        y = np.arange(1, len(v) + 1) / len(v)
    else:
        w = np.asarray(weights, dtype=float)[mask][order]
        y = np.cumsum(w) / np.sum(w)
    ax.plot(v, y * 100, label=label, color=color, linewidth=1.6, **kw)


def load_outputs(outdir: Path):
    pairs = pd.read_parquet(outdir / "eligible_pairs.parquet")
    retain = pd.read_parquet(outdir / "retain_step_results.parquet")
    strip = pd.read_parquet(outdir / "strip_step_results.parquet")
    summary_retain = pd.read_csv(outdir / "summary_retain.csv")
    sens_retain = pd.read_csv(outdir / "sensitivity_retain.csv")
    sens_queue = pd.read_csv(outdir / "sensitivity_prefill_queue.csv")
    request = pd.read_csv(outdir / "request_level_summary.csv")
    return pairs, retain, strip, summary_retain, sens_retain, sens_queue, request


# ---------------------------------------------------------------------------
# Retain plots
# ---------------------------------------------------------------------------

def plot_retain_overlap(retain, summary, networks, outdir):
    main_bw = networks["main"]["bandwidth_gbps"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for i, m in enumerate(MODEL_ORDER):
        sub = retain[(retain["model"] == m) & (retain["bandwidth_gbps"] == main_bw)]
        _cdf(ax, sub["transfer_ms"], MODEL_LABELS[m], PLOT_COLORS[i])
        _cdf(ax, sub["hidden_transfer_ms"], None, PLOT_COLORS[i], linestyle="--", alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("D→P delta-KV transfer time (ms)")
    ax.set_ylabel("CDF (% of tool-initiated steps)")
    ax.set_title(f"Transfer time CDF @ {main_bw} GB/s\n(solid: total, dashed: hidden part)")
    ax.legend(fontsize=8)
    polish_axes(ax)

    ax = axes[1]
    mb = summary[(summary["group_type"] == "model_bandwidth")
                 & (summary["variant"] == "unfiltered")]
    for i, m in enumerate(MODEL_ORDER):
        sub = mb[mb["model"] == m].sort_values("bandwidth_gbps")
        ax.plot(sub["bandwidth_gbps"], sub["fully_hidden_frac"] * 100, "o-",
                label=MODEL_LABELS[m], color=PLOT_COLORS[i])
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 900])
    ax.set_xticklabels(["25", "50", "100", "200", "900"])
    ax.set_xlabel("Effective bandwidth (GB/s)")
    ax.set_ylabel("Fully hidden steps (%)")
    ax.set_title("Complete-overlap fraction vs bandwidth")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    polish_axes(ax)

    ax = axes[2]
    tc = summary[(summary["group_type"] == "tool_category")
                 & (summary["variant"] == "unfiltered")
                 & (summary["model"] == MODEL_ORDER[0])].sort_values("fully_hidden_frac")
    ax.barh(tc["group"], tc["fully_hidden_frac"] * 100, color=plot_color("codex", 0))
    ax.set_xlabel("Fully hidden steps (%)")
    ax.set_title(f"By tool type ({MODEL_LABELS[MODEL_ORDER[0]].split(' (')[0]},"
                 f" {main_bw} GB/s)")
    ax.set_xlim(0, 100)
    polish_axes(ax, grid_axis="x")

    fig.suptitle("Retain-thinking: proactive D→P delta-KV handoff overlap",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_plot(fig, outdir / "retain_overlap_cdf.png")


def plot_retain_exposed(retain, pairs, models, networks, outdir):
    main_bw = networks["main"]["bandwidth_gbps"]
    byte_w = {}
    for m in MODEL_ORDER:
        mk = models[m]
        byte_w[m] = pd.Series(
            km.transfer_bytes(pairs["output_tokens"].to_numpy(dtype=float),
                              mk.logical_kv_bytes_per_token,
                              mk.block_size_tokens, mk.physical_transfer_multiplier),
            index=pairs["pair_id"].to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for panel, (ax, weighted) in enumerate(zip(axes, [False, True])):
        for i, m in enumerate(MODEL_ORDER):
            for j, bw in enumerate([25, main_bw, 900]):
                sub = retain[(retain["model"] == m) & (retain["bandwidth_gbps"] == bw)]
                w = byte_w[m].reindex(sub["pair_id"]).to_numpy() if weighted else None
                _cdf(ax, sub["exposed_transfer_ms"].clip(lower=1e-3),
                     f"{m.split('-')[0]} @{bw}GB/s" if panel == 0 or True else None,
                     PLOT_COLORS[i], weights=w,
                     linestyle=["-", "--", ":"][j])
        ax.set_xscale("log")
        ax.set_xlabel("Exposed D→P transfer time (ms, 0 shown at 1e-3)")
        ax.set_ylabel("CDF (%)" if not weighted else "byte-weighted CDF (%)")
        ax.set_title("Step-weighted" if not weighted else "Byte-weighted")
        ax.set_ylim(0, 100.5)
        if panel == 0:
            ax.legend(fontsize=6.5, ncol=1)
        polish_axes(ax)
    fig.suptitle("Retain-thinking: exposed (critical-path) transfer time",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_plot(fig, outdir / "retain_exposed_latency_cdf.png")


def plot_retain_heatmap(pairs, models, networks, outdir):
    main_oh = networks["main"]["fixed_overhead_ms"]
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(15, 4.4), sharey=True)
    x = np.log2(np.clip(pairs["output_tokens"], 1, None))
    y = np.log10(np.clip(pairs["slack_ms"], 0.1, None))
    xbins = np.linspace(0, 15, 60)
    ybins = np.linspace(-1, 6.5, 60)
    for ax, m in zip(axes, MODEL_ORDER):
        mk = models[m]
        h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
        pcm = ax.pcolormesh(xbins, ybins, np.log10(h.T + 1), cmap="Blues")
        for bw, color, style in [(25, "#dc2626", "--"), (100, "#111827", "-"),
                                 (900, "#059669", ":")]:
            tok = np.logspace(0, np.log10(2 ** 15), 200)
            t_ms = km.transfer_ms(tok, bytes_per_token=mk.logical_kv_bytes_per_token,
                                  block_size_tokens=mk.block_size_tokens,
                                  bandwidth_gbps=bw, fixed_overhead_ms=main_oh)
            ax.plot(np.log2(tok), np.log10(t_ms), color=color, linestyle=style,
                    linewidth=1.4, label=f"boundary @{bw} GB/s")
        ax.set_title(MODEL_LABELS[m], fontsize=9)
        ax.set_xlabel("output tokens (G_i = R+A)")
        ax.set_xticks([0, 4, 7, 10, 13, 15])
        ax.set_xticklabels(["1", "16", "128", "1K", "8K", "32K"])
        polish_axes(ax)
    axes[0].set_ylabel("tool slack (ms)")
    axes[0].set_yticks([-1, 0, 1, 2, 3, 4, 5, 6])
    axes[0].set_yticklabels(["0.1", "1", "10", "100", "1s", "10s", "100s", "1000s"])
    axes[0].legend(fontsize=7, loc="upper left")
    fig.colorbar(pcm, ax=axes, shrink=0.85, label="log10(steps+1)")
    fig.suptitle("Output tokens × tool slack — points above the boundary are fully hidden",
                 fontweight="semibold")
    save_plot(fig, outdir / "retain_output_vs_slack_heatmap.png")


def plot_sensitivity_network(summary, sens, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]  # model x bandwidth matrix (fully hidden %)
    mb = summary[(summary["group_type"] == "model_bandwidth")
                 & (summary["variant"] == "unfiltered")]
    piv = mb.pivot_table(index="model", columns="bandwidth_gbps",
                         values="fully_hidden_frac").reindex(MODEL_ORDER) * 100
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{int(c)}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [m.split("-")[0] for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("bandwidth (GB/s)")
    ax.set_title("Fully hidden % (model × network)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]  # fixed overhead sweep
    net = sens[sens["sweep"] == "network"]
    for i, m in enumerate(MODEL_ORDER):
        for j, oh in enumerate(sorted(net["fixed_overhead_ms"].unique())):
            sub = net[(net["model"] == m) & (net["fixed_overhead_ms"] == oh)].sort_values(
                "bandwidth_gbps")
            ax.plot(sub["bandwidth_gbps"], sub["fully_hidden_frac"] * 100,
                    color=PLOT_COLORS[i], alpha=0.35 + 0.16 * j,
                    linestyle=["-", "--", "-.", ":"][j],
                    label=f"{m.split('-')[0]} oh={oh}ms" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("bandwidth (GB/s)")
    ax.set_ylabel("fully hidden (%)")
    ax.set_title("Fixed-overhead sensitivity (line style)\ncolor = model")
    ax.legend(fontsize=6.5)
    polish_axes(ax)

    ax = axes[2]  # kv size sweep + dtype/tp
    sweep = sens[sens["sweep"] == "kv_size_sweep"]
    for i, m in enumerate(sorted(sweep["model"].unique(),
                                 key=lambda s: int(s[2:-3]))):
        sub = sweep[sweep["model"] == m].sort_values("bandwidth_gbps")
        ax.plot(sub["bandwidth_gbps"], sub["fully_hidden_frac"] * 100, "o-",
                label=f"KV {m[2:]}/tok", color=PLOT_COLORS[i % len(PLOT_COLORS)])
    ax.set_xscale("log")
    ax.set_xlabel("bandwidth (GB/s)")
    ax.set_ylabel("fully hidden (%)")
    ax.set_title("Architecture-independent KV-size sweep\n(Kimi-K3 uncertainty band)")
    ax.legend(fontsize=7)
    polish_axes(ax)

    fig.suptitle("Retain-thinking sensitivity: network, overhead, KV size",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_plot(fig, outdir / "sensitivity_network.png")


# ---------------------------------------------------------------------------
# Strip plots
# ---------------------------------------------------------------------------

def plot_strip_required_throughput(strip, pairs, outdir):
    meta = pairs.set_index("pair_id")
    base = strip.drop_duplicates("pair_id").set_index("pair_id")
    base = base.join(meta[["prefix_tokens", "provider", "model"]], rsuffix="_m")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    codex = base[base["visible_kind"] == "exact"]
    claude = base[base["visible_kind"] == "upper_bound"]
    req_c = codex["required_prefill_tokens_per_sec"]
    _cdf(ax, req_c[np.isfinite(req_c)].clip(lower=1e-1), "Codex (exact visible A)",
         plot_color("codex", 0))
    req_cl = claude["required_prefill_tokens_per_sec"]
    _cdf(ax, req_cl[np.isfinite(req_cl)].clip(lower=1e-1),
         "Claude (upper bound: A = full output)", plot_color("claude", 1))
    ax.set_xscale("log")
    ax.set_xlabel("required prefill throughput to fully hide A (tokens/s)")
    ax.set_ylabel("CDF (% of steps with slack > 0)")
    ax.set_title("Trace-only: required append-prefill rate")
    ax.legend(fontsize=8)
    polish_axes(ax)

    ax = axes[1]
    edges = [0, 16_384, 65_536, 131_072, 262_144, 1 << 40]
    labels = ["<16K", "16–64K", "64–128K", "128–256K", "≥256K"]
    codex = codex.assign(prefix_bin=pd.cut(codex["prefix_tokens"], bins=edges,
                                           labels=labels, right=False))
    for i, b in enumerate(labels):
        sub = codex[codex["prefix_bin"] == b]["required_prefill_tokens_per_sec"]
        _cdf(ax, sub[np.isfinite(sub)].clip(lower=1e-1), f"prefix {b}",
             PLOT_COLORS[i % len(PLOT_COLORS)])
    ax.set_xscale("log")
    ax.set_xlabel("required prefill throughput (tokens/s)")
    ax.set_title("Codex, by prefix-length bin")
    ax.legend(fontsize=7.5)
    polish_axes(ax)

    fig.suptitle("Strip-thinking Phase 1 (trace-only): |A| / canonicalization slack",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_plot(fig, outdir / "strip_required_throughput_cdf.png")


def plot_strip_hidden_fraction(strip, sens_queue, prefill_cfg, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for i, m in enumerate(MODEL_ORDER):
        sub = strip[strip["model"] == m]
        shadow = sub["shadow_prefill_ms"].to_numpy()
        exposed = sub["exposed_shadow_ms"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            hidden_frac = np.where(shadow > 0, 1 - exposed / shadow, 1.0)
        _cdf(ax, hidden_frac * 100, MODEL_LABELS[m], PLOT_COLORS[i])
    ax.set_xlabel("hidden fraction of shadow prefill F(C, A) (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_title(f"Per-step hidden fraction ({prefill_cfg['main_profile']})")
    ax.legend(fontsize=7.5, loc="upper left")
    polish_axes(ax)

    ax = axes[1]
    q0 = sens_queue[sens_queue["p_queue_ms"] == 0]
    piv = q0.pivot_table(index="model", columns="profile",
                         values="shadow_fully_hidden_frac").reindex(MODEL_ORDER) * 100
    xpos = np.arange(len(piv.index))
    width = 0.8 / len(piv.columns)
    for j, prof in enumerate(piv.columns):
        ax.bar(xpos + j * width, piv[prof], width, label=prof,
               color=PLOT_COLORS[j % len(PLOT_COLORS)])
    ax.set_xticks(xpos + width, [m.split("-")[0] for m in piv.index])
    ax.set_ylabel("fully hidden shadow prefill (%)")
    ax.set_title("By prefill profile (q=0)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5)
    polish_axes(ax, grid_axis="y")

    fig.suptitle("Strip-thinking: how much of the A-prefill hides under tool execution",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_plot(fig, outdir / "strip_hidden_prefill_fraction_cdf.png")


def plot_strip_speedup(strip, pairs, request, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]  # S0/S1/S2 step latency comparison (medians + p90)
    stats = []
    for m in MODEL_ORDER:
        sub = strip[(strip["model"] == m)].dropna(subset=["T_S0_ms"])
        stats.append([sub[c].median() for c in ["T_S0_ms", "T_S1_ms", "T_S2_ms"]])
    stats = np.asarray(stats)
    xpos = np.arange(len(MODEL_ORDER))
    for j, (label, color) in enumerate(zip(["S0 reactive", "S1 shadow prefill",
                                            "S2 shadow + pre-copy"], PLOT_COLORS)):
        ax.bar(xpos + j * 0.26, stats[:, j], 0.26, label=label, color=color)
    ax.set_xticks(xpos + 0.26, [m.split("-")[0] for m in MODEL_ORDER])
    ax.set_ylabel("median post-tool critical path (ms)")
    ax.set_title("S0 / S1 / S2 step latency (median)")
    ax.legend(fontsize=7.5)
    polish_axes(ax, grid_axis="y")

    ax = axes[1]  # absolute saving CDFs
    for i, m in enumerate(MODEL_ORDER):
        sub = strip[strip["model"] == m]
        _cdf(ax, sub["saving_S1_ms"].dropna().clip(lower=1e-2), f"{m.split('-')[0]} S1",
             PLOT_COLORS[i])
        _cdf(ax, sub["saving_S2_ms"].dropna().clip(lower=1e-2), f"{m.split('-')[0]} S2",
             PLOT_COLORS[i], linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("per-step saving vs S0 (ms)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Step-level absolute saving")
    ax.legend(fontsize=6.5)
    polish_axes(ax)

    ax = axes[2]  # request-level absolute + relative saving
    m = MODEL_ORDER[0]
    col = f"strip_saving_S1_ms_{m}"
    if col in request.columns:
        sav = request[col].dropna()
        _cdf(ax, sav.clip(lower=1e-2), "absolute (ms)", PLOT_COLORS[0])
        ax.set_xscale("log")
        ax.set_xlabel("request-level S1 saving (ms)")
        ax.set_ylabel("CDF (%)")
        ax2 = ax.twiny()
        rel = (request[col] / (request["response_time_ms"]
                               + request[col].clip(lower=0))).dropna()
        _cdf(ax2, rel.clip(lower=1e-5) * 100, "relative (%)", PLOT_COLORS[3])
        ax2.set_xscale("log")
        ax2.set_xlabel("relative saving (% of response time)", color=PLOT_COLORS[3])
        ax2.tick_params(axis="x", colors=PLOT_COLORS[3])
        ax.set_title(f"Request-level saving ({m.split('-')[0]})")
        ax.legend(fontsize=7.5, loc="upper left")
    polish_axes(ax)

    fig.suptitle("Strip-thinking: S0 vs S1 vs S2 and end-to-end savings",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_plot(fig, outdir / "strip_speedup_cdf.png")


def plot_strip_heatmap(strip, pairs, models, profiles, prefill_cfg, outdir):
    meta = pairs.set_index("pair_id")[["prefix_tokens", "visible_output_tokens",
                                       "canon_slack_ms", "visible_is_upper_bound"]]
    codex = meta[~meta["visible_is_upper_bound"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    x = np.log2(np.clip(codex["visible_output_tokens"], 1, None))
    y = np.log10(np.clip(codex["canon_slack_ms"], 0.1, None))
    xbins = np.linspace(0, 14, 56)
    ybins = np.linspace(-1, 6.5, 56)
    m = models[MODEL_ORDER[0]]
    profile = profiles[prefill_cfg["main_profile"]]
    for ax, prefix_tok, label in zip(axes, [16_384, 131_072, 524_288],
                                     ["16K", "128K", "512K"]):
        h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
        pcm = ax.pcolormesh(xbins, ybins, np.log10(h.T + 1), cmap="Purples")
        tok = np.logspace(0, np.log10(2 ** 14), 200)
        t_ms = km.prefill_ms_roofline(prefix_tok, tok, model=m, profile=profile)
        ax.plot(np.log2(tok), np.log10(np.maximum(t_ms, 0.1)), color="#111827",
                linewidth=1.5, label=f"F(C={label}, A) boundary")
        ax.set_title(f"prefix C = {label} tokens")
        ax.set_xlabel("visible assistant output |A| (tokens)")
        ax.set_xticks([0, 4, 7, 10, 13])
        ax.set_xticklabels(["1", "16", "128", "1K", "8K"])
        ax.legend(fontsize=7.5, loc="upper left")
        polish_axes(ax)
    axes[0].set_ylabel("canonicalization slack (ms)")
    axes[0].set_yticks([-1, 0, 1, 2, 3, 4, 5, 6])
    axes[0].set_yticklabels(["0.1", "1", "10", "100", "1s", "10s", "100s", "1000s"])
    fig.colorbar(pcm, ax=axes, shrink=0.85, label="log10(steps+1)")
    fig.suptitle(
        f"Codex steps: |A| × canonicalization slack; boundary = shadow-prefill time "
        f"({MODEL_ORDER[0]}, {prefill_cfg['main_profile']}); above = fully hidden",
        fontweight="semibold")
    save_plot(fig, outdir / "strip_prefix_output_slack_heatmap.png")


def plot_sensitivity_prefill(strip, sens_queue, summary_strip, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]  # queue sweep
    for i, m in enumerate(MODEL_ORDER):
        for j, prof in enumerate(sorted(sens_queue["profile"].unique())):
            sub = sens_queue[(sens_queue["model"] == m)
                             & (sens_queue["profile"] == prof)].sort_values("p_queue_ms")
            ax.plot(sub["p_queue_ms"] + 0.1, sub["shadow_fully_hidden_frac"] * 100,
                    "o-", color=PLOT_COLORS[i], alpha=[1.0, 0.6, 0.35][j % 3],
                    label=f"{m.split('-')[0]} / {prof.split('_')[-1]}" if True else None)
    ax.set_xscale("log")
    ax.set_xlabel("P-node queue latency (ms, +0.1 offset)")
    ax.set_ylabel("fully hidden shadow prefill (%)")
    ax.set_title("Queueing sensitivity\n(proactive queue hides in slack)")
    ax.legend(fontsize=6)
    polish_axes(ax)

    ax = axes[1]  # saving by prefix bin
    pb = summary_strip[(summary_strip["group_type"] == "prefix_bin")]
    for i, m in enumerate(MODEL_ORDER):
        sub = pb[pb["model"] == m]
        order = ["<16K", "16-64K", "64-128K", "128-256K", ">=256K"]
        sub = sub.set_index("group").reindex(order)
        ax.plot(range(len(sub)), sub["saving_S1_p50_ms"], "o-",
                color=PLOT_COLORS[i], label=f"{m.split('-')[0]} S1")
        ax.plot(range(len(sub)), sub["saving_S2_p50_ms"], "s--",
                color=PLOT_COLORS[i], alpha=0.55, label=f"{m.split('-')[0]} S2")
    ax.set_xticks(range(5), ["<16K", "16–64K", "64–128K", "128–256K", "≥256K"],
                  fontsize=8)
    ax.set_xlabel("prefix-length bin")
    ax.set_ylabel("median saving vs S0 (ms)")
    ax.set_yscale("log")
    ax.set_title("Median S1/S2 saving by prefix length")
    ax.legend(fontsize=6.5)
    polish_axes(ax)

    ax = axes[2]  # saving by tool category
    tc = summary_strip[(summary_strip["group_type"] == "tool_category")
                       & (summary_strip["model"] == MODEL_ORDER[0])]
    tc = tc.sort_values("saving_S1_p50_ms")
    ax.barh(tc["group"], tc["saving_S1_p50_ms"], color=PLOT_COLORS[0], label="S1")
    ax.barh(tc["group"], tc["saving_S2_p50_ms"], height=0.45,
            color=PLOT_COLORS[2], label="S2")
    ax.set_xlabel("median saving vs S0 (ms)")
    ax.set_title(f"By tool type ({MODEL_ORDER[0].split('-')[0]})")
    ax.legend(fontsize=7.5)
    polish_axes(ax, grid_axis="x")

    fig.suptitle("Strip-thinking sensitivity: queueing, prefix length, tool type",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_plot(fig, outdir / "sensitivity_prefill.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=EXP_DIR / "outputs")
    args = parser.parse_args()
    outdir = args.output_dir

    models = km.load_models()
    networks = km.load_networks()
    prefill_cfg = km.load_prefill_profiles()
    profiles = km.parse_profiles(prefill_cfg)

    pairs, retain, strip, summary_retain, sens_retain, sens_queue, request = \
        load_outputs(outdir)
    summary_strip = pd.read_csv(outdir / "summary_strip.csv")

    plot_retain_overlap(retain, summary_retain, networks, outdir)
    plot_retain_exposed(retain, pairs, models, networks, outdir)
    plot_retain_heatmap(pairs, models, networks, outdir)
    plot_sensitivity_network(summary_retain, sens_retain, outdir)
    plot_strip_required_throughput(strip, pairs, outdir)
    plot_strip_hidden_fraction(strip, sens_queue, prefill_cfg, outdir)
    plot_strip_speedup(strip, pairs, request, outdir)
    plot_strip_heatmap(strip, pairs, models, profiles, prefill_cfg, outdir)
    plot_sensitivity_prefill(strip, sens_queue, summary_strip, outdir)

    png_sidecar.make_self_contained(
        outdir,
        code_files=[Path(__file__), EXP_DIR / "analyze.py", EXP_DIR / "kv_models.py",
                    *png_sidecar.util_code_files()],
        readme_path=EXP_DIR / "README.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
