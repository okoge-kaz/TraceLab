"""Pure model code for the KV-overlap replay: transfer model, prefill model,
visible-output accounting, and pair-classification predicates.

Kept free of DuckDB/pandas so the unit tests in tests/ can exercise every
formula directly. All array arguments accept numpy arrays or scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelKV:
    name: str
    kv_layout: str
    kv_dtype: str
    logical_kv_bytes_per_token: int
    physical_transfer_multiplier: float
    block_size_tokens: int
    active_params: float
    attn_flops_per_ctx_token: float
    kv_bytes_by_dtype: dict[str, int]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_models(config_dir: Path = CONFIG_DIR) -> dict[str, ModelKV]:
    raw = load_yaml(config_dir / "models.yaml")
    models: dict[str, ModelKV] = {}
    for name, cfg in raw["models"].items():
        models[name] = ModelKV(
            name=name,
            kv_layout=cfg["kv_layout"],
            kv_dtype=cfg["kv_dtype"],
            logical_kv_bytes_per_token=int(cfg["logical_kv_bytes_per_token"]),
            physical_transfer_multiplier=float(cfg["physical_transfer_multiplier"]),
            block_size_tokens=int(cfg["block_size_tokens"]),
            active_params=float(cfg["active_params"]),
            attn_flops_per_ctx_token=float(cfg["attn_flops_per_ctx_token"]),
            kv_bytes_by_dtype={k: int(v) for k, v in cfg["kv_bytes_per_token_by_dtype"].items()},
        )
    return models


def load_kv_size_sweep(config_dir: Path = CONFIG_DIR) -> tuple[int, list[int]]:
    raw = load_yaml(config_dir / "models.yaml")["kv_size_sweep"]
    return int(raw["block_size_tokens"]), [int(v) for v in raw["bytes_per_token"]]


def load_networks(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    return load_yaml(config_dir / "networks.yaml")


def load_prefill_profiles(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    return load_yaml(config_dir / "prefill_profiles.yaml")


# ---------------------------------------------------------------------------
# Transfer model (Retain-thinking D->P delta-KV handoff; also P->D transfers
# in strip scenarios)
# ---------------------------------------------------------------------------

def rounded_tokens(delta_kv_tokens, block_size_tokens: int):
    """KV block rounding: ceil(tokens / block) * block (0 stays 0)."""
    tokens = np.asarray(delta_kv_tokens, dtype=np.float64)
    return np.ceil(tokens / block_size_tokens) * block_size_tokens


def transfer_bytes(delta_kv_tokens, bytes_per_token: float, block_size_tokens: int,
                   physical_transfer_multiplier: float = 1.0):
    return (
        rounded_tokens(delta_kv_tokens, block_size_tokens)
        * bytes_per_token
        * physical_transfer_multiplier
    )


def transfer_ms(delta_kv_tokens, *, bytes_per_token: float, block_size_tokens: int,
                bandwidth_gbps: float, fixed_overhead_ms: float,
                physical_transfer_multiplier: float = 1.0):
    """Wire time of one KV transfer. bandwidth_gbps is GB/s (1e9 bytes/s)."""
    tbytes = transfer_bytes(
        delta_kv_tokens, bytes_per_token, block_size_tokens, physical_transfer_multiplier
    )
    return fixed_overhead_ms + 1000.0 * tbytes / (bandwidth_gbps * 1e9)


def overlap_split(transfer_time_ms, slack_ms):
    """Return (hidden_ms, exposed_ms, fully_hidden, overlap_fraction)."""
    transfer_time_ms = np.asarray(transfer_time_ms, dtype=np.float64)
    slack_ms = np.asarray(slack_ms, dtype=np.float64)
    hidden = np.minimum(slack_ms, transfer_time_ms)
    exposed = np.maximum(0.0, transfer_time_ms - slack_ms)
    fully_hidden = transfer_time_ms <= slack_ms
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(transfer_time_ms > 0, hidden / transfer_time_ms, 1.0)
    return hidden, exposed, fully_hidden, frac


def clip_slack_ms(ready_us, start_us):
    """slack = max(0, t_ready - t_start) in ms from epoch-us ints (NaN-safe)."""
    ready = np.asarray(ready_us, dtype=np.float64)
    start = np.asarray(start_us, dtype=np.float64)
    return np.maximum(0.0, (ready - start) / 1000.0)


# ---------------------------------------------------------------------------
# Visible assistant output tokens
# ---------------------------------------------------------------------------

def codex_visible_output_tokens(output_tokens, reasoning_output_tokens):
    """Codex: exact visible output = max(0, output - reasoning)."""
    out = np.asarray(output_tokens, dtype=np.float64)
    reasoning = np.nan_to_num(np.asarray(reasoning_output_tokens, dtype=np.float64), nan=0.0)
    return np.maximum(0.0, out - reasoning)


# ---------------------------------------------------------------------------
# Prefill model (Phase 2 synthetic profiles / lookup table)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefillProfile:
    name: str
    tp_size: int
    per_gpu_flops: float
    mfu: float
    launch_overhead_ms: float
    concurrency_penalty: dict[int, float]

    def effective_flops(self, concurrency: int = 1) -> float:
        penalty = self.concurrency_penalty.get(int(concurrency), 1.0)
        return self.tp_size * self.per_gpu_flops * self.mfu / penalty


def parse_profiles(raw: dict[str, Any]) -> dict[str, PrefillProfile]:
    profiles = {}
    for name, cfg in raw["profiles"].items():
        profiles[name] = PrefillProfile(
            name=name,
            tp_size=int(cfg["tp_size"]),
            per_gpu_flops=float(cfg["per_gpu_flops"]),
            mfu=float(cfg["mfu"]),
            launch_overhead_ms=float(cfg["launch_overhead_ms"]),
            concurrency_penalty={int(k): float(v) for k, v in cfg["concurrency_penalty"].items()},
        )
    return profiles


def prefill_ms_roofline(prefix_tokens, suffix_tokens, *, model: ModelKV,
                        profile: PrefillProfile, concurrency: int = 1):
    """Synthetic 2D append-prefill time: teacher-forced prefill of `suffix`
    tokens on top of a resident `prefix` KV.

    prefill_ms(P, S) = overhead + S*linear/F + (P*S + S^2/2)*attn/F
    """
    prefix = np.asarray(prefix_tokens, dtype=np.float64)
    suffix = np.asarray(suffix_tokens, dtype=np.float64)
    f_eff = profile.effective_flops(concurrency)
    linear_flops = suffix * 2.0 * model.active_params
    attn_flops = (prefix * suffix + suffix * suffix / 2.0) * model.attn_flops_per_ctx_token
    time_ms = profile.launch_overhead_ms + (linear_flops + attn_flops) / f_eff * 1e3
    return np.where(suffix <= 0, 0.0, time_ms)


class PrefillLookupTable:
    """Bilinear (in log-token space) interpolation over a measured grid.

    Table rows: (prefix_tokens, suffix_tokens) -> prefill_ms for a fixed
    (model, concurrency, tp) slice. Values outside the grid clamp to the edge.
    """

    def __init__(self, prefix_grid: list[float], suffix_grid: list[float],
                 time_ms_grid: np.ndarray):
        self.prefix_grid = np.asarray(sorted(prefix_grid), dtype=np.float64)
        self.suffix_grid = np.asarray(sorted(suffix_grid), dtype=np.float64)
        self.time_ms = np.asarray(time_ms_grid, dtype=np.float64)
        if self.time_ms.shape != (len(self.prefix_grid), len(self.suffix_grid)):
            raise ValueError("time_ms_grid shape mismatch")

    @staticmethod
    def _axis_weights(grid: np.ndarray, values: np.ndarray):
        logv = np.log2(np.clip(values, grid[0], grid[-1]))
        logg = np.log2(grid)
        hi = np.clip(np.searchsorted(logg, logv, side="right"), 1, len(grid) - 1)
        lo = hi - 1
        span = logg[hi] - logg[lo]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(span > 0, (logv - logg[lo]) / span, 0.0)
        return lo, hi, w

    def __call__(self, prefix_tokens, suffix_tokens):
        prefix = np.atleast_1d(np.asarray(prefix_tokens, dtype=np.float64))
        suffix = np.atleast_1d(np.asarray(suffix_tokens, dtype=np.float64))
        pl, ph, pw = self._axis_weights(self.prefix_grid, np.maximum(prefix, 1.0))
        sl, sh, sw = self._axis_weights(self.suffix_grid, np.maximum(suffix, 1.0))
        t = (
            self.time_ms[pl, sl] * (1 - pw) * (1 - sw)
            + self.time_ms[ph, sl] * pw * (1 - sw)
            + self.time_ms[pl, sh] * (1 - pw) * sw
            + self.time_ms[ph, sh] * pw * sw
        )
        return np.where(suffix <= 0, 0.0, t)


def table_from_profile(model: ModelKV, profile: PrefillProfile, *,
                       prefix_grid: list[int], suffix_grid: list[int],
                       concurrency: int = 1) -> PrefillLookupTable:
    prefix = np.asarray(prefix_grid, dtype=np.float64)
    suffix = np.asarray(suffix_grid, dtype=np.float64)
    grid = prefill_ms_roofline(
        prefix[:, None], suffix[None, :], model=model, profile=profile,
        concurrency=concurrency,
    )
    return PrefillLookupTable(list(prefix), list(suffix), grid)


# ---------------------------------------------------------------------------
# Strip-thinking scenarios (S0 reactive / S1 shadow / S2 shadow + pre-copy)
# ---------------------------------------------------------------------------

def strip_scenarios(*, prefix_tokens, visible_tokens, tool_append_tokens,
                    canon_slack_ms, prefill, xfer, q_reactive_ms=0.0,
                    q_proactive_ms=0.0):
    """Post-tool critical-path time for S0/S1/S2.

    prefill(P, S) -> ms; xfer(tokens) -> ms (network transfer incl. fixed overhead).
    All token args may be numpy arrays. Returns dict of arrays.
    """
    P = np.asarray(prefix_tokens, dtype=np.float64)
    A = np.asarray(visible_tokens, dtype=np.float64)
    O = np.asarray(tool_append_tokens, dtype=np.float64)
    slack = np.asarray(canon_slack_ms, dtype=np.float64)

    f_c_a = prefill(P, A)                # F(C_i, A_i)
    f_ca_o = prefill(P + A, O)           # F(C_i || A_i, O_i)
    f_c_ao = prefill(P, A + O)           # F(C_i, A_i || O_i)

    # S0: everything after the tool result.
    t_s0 = q_reactive_ms + f_c_ao + xfer(A + O)

    # S1: shadow-prefill A during tool execution; transfer A||O after.
    shadow_stage = q_proactive_ms + f_c_a
    exposed_shadow = np.maximum(0.0, shadow_stage - slack)
    t_s1 = exposed_shadow + f_ca_o + xfer(A + O)

    # S2: shadow-prefill A and pre-copy KV(A); only O prefill+transfer after.
    shadow_precopy = q_proactive_ms + f_c_a + xfer(A)
    exposed_precopy = np.maximum(0.0, shadow_precopy - slack)
    t_s2 = exposed_precopy + f_ca_o + xfer(O)

    return {
        "T_S0": t_s0,
        "T_S1": t_s1,
        "T_S2": t_s2,
        "exposed_shadow_ms": exposed_shadow,
        "exposed_shadow_precopy_ms": exposed_precopy,
        "shadow_prefill_ms": f_c_a,
        "saving_S1": t_s0 - t_s1,
        "saving_S2": t_s0 - t_s2,
    }


# ---------------------------------------------------------------------------
# Pair classification (output-cached-like vs output-resend-like), following
# artifacts/llm_generation/output_append_assignment tolerance predicates.
# ---------------------------------------------------------------------------

TOLERANCE_ABSOLUTE_TOKENS = 512.0
TOLERANCE_RELATIVE_FRACTION = 0.10


def _tolerance(prev_output):
    prev_output = np.asarray(prev_output, dtype=np.float64)
    return np.maximum(TOLERANCE_ABSOLUTE_TOKENS, TOLERANCE_RELATIVE_FRACTION * prev_output)


def classify_output_assignment(prefix_delta, next_append, prev_output):
    """Per-pair label: 'output_cached_like' | 'output_resend_like' | 'ambiguous'.

    prefix_delta = next.prefix_tokens - prev.input_tokens_total.
    cached-like: the prior output shows up as prefix growth (KV kept server-side).
    resend-like: prefix rejects the output but next append can contain it.
    """
    prefix_delta = np.asarray(prefix_delta, dtype=np.float64)
    next_append = np.asarray(next_append, dtype=np.float64)
    prev_output = np.asarray(prev_output, dtype=np.float64)
    tol = _tolerance(prev_output)
    prefix_close = np.abs(prefix_delta - prev_output) <= tol
    prefix_reject = prefix_delta < prev_output - tol
    append_can = next_append >= prev_output - tol
    labels = np.full(prefix_delta.shape, "ambiguous", dtype=object)
    labels[prefix_close] = "output_cached_like"
    labels[prefix_reject & append_can & ~prefix_close] = "output_resend_like"
    return labels


def estimated_tool_append_tokens(labels, next_append, visible_output):
    """Section-14 estimate of the next round's tool-result append size.

    resend-like: next_append - visible_output (the visible output is resent, the
    rest of the append is framing + tool results); cached-like: next_append.
    Named *_append_* deliberately: this includes message framing / cache-boundary
    effects and is NOT a raw tool-output token count. Ambiguous pairs -> NaN.
    """
    next_append = np.asarray(next_append, dtype=np.float64)
    visible = np.asarray(visible_output, dtype=np.float64)
    est = np.where(
        labels == "output_resend_like",
        next_append - visible,
        next_append,
    )
    est = np.where(labels == "ambiguous", np.nan, est)
    return est
