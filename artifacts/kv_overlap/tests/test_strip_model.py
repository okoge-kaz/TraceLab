"""Unit tests for the strip-thinking (assistant-output canonicalization) model."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kv_models as km


def test_codex_visible_output_tokens():
    np.testing.assert_allclose(
        km.codex_visible_output_tokens(np.array([100, 50, 30]),
                                       np.array([40, 60, np.nan])),
        [60.0, 0.0, 30.0],  # clamped at 0; NaN reasoning -> treated as 0
    )


def _linear_prefill(per_token_ms=0.01, overhead_ms=1.0):
    def f(prefix, suffix):
        suffix = np.asarray(suffix, dtype=float)
        return np.where(suffix <= 0, 0.0, overhead_ms + suffix * per_token_ms)
    return f


def _fixed_xfer(fixed_ms=1.0, per_token_ms=0.001):
    def x(tokens):
        return fixed_ms + np.asarray(tokens, dtype=float) * per_token_ms
    return x


def test_strip_s1_saving_with_ample_slack():
    # A=1000 tok, O=100 tok, slack 1e6 ms: S1 hides F(C,A) entirely.
    res = km.strip_scenarios(
        prefix_tokens=np.array([10000.0]), visible_tokens=np.array([1000.0]),
        tool_append_tokens=np.array([100.0]), canon_slack_ms=np.array([1e6]),
        prefill=_linear_prefill(), xfer=_fixed_xfer())
    # S0 = F(C, A+O) + X(A+O) = 1 + 11 + (1 + 1.1) = 14.1
    assert abs(res["T_S0"][0] - 14.1) < 1e-9
    # S1 = 0 exposed + F(C||A, O) + X(A+O) = (1+1) + 2.1 = 4.1
    assert abs(res["T_S1"][0] - 4.1) < 1e-9
    assert abs(res["saving_S1"][0] - 10.0) < 1e-9


def test_strip_s2_double_fixed_overhead_penalty_short_slack():
    # Zero slack: S2 pays the pre-copy X(A) fully exposed plus a second fixed
    # transfer overhead for O -> S2 must be worse than S1 here.
    res = km.strip_scenarios(
        prefix_tokens=np.array([10000.0]), visible_tokens=np.array([1000.0]),
        tool_append_tokens=np.array([100.0]), canon_slack_ms=np.array([0.0]),
        prefill=_linear_prefill(), xfer=_fixed_xfer(fixed_ms=5.0))
    assert res["T_S2"][0] > res["T_S1"][0]
    # And with zero slack S1's shadow stage is fully exposed:
    assert res["exposed_shadow_ms"][0] == res["shadow_prefill_ms"][0]


def test_strip_queue_hidden_proactive_exposed_reactive():
    base = km.strip_scenarios(
        prefix_tokens=np.array([1000.0]), visible_tokens=np.array([100.0]),
        tool_append_tokens=np.array([50.0]), canon_slack_ms=np.array([1e6]),
        prefill=_linear_prefill(), xfer=_fixed_xfer())
    queued = km.strip_scenarios(
        prefix_tokens=np.array([1000.0]), visible_tokens=np.array([100.0]),
        tool_append_tokens=np.array([50.0]), canon_slack_ms=np.array([1e6]),
        prefill=_linear_prefill(), xfer=_fixed_xfer(),
        q_reactive_ms=50.0, q_proactive_ms=50.0)
    # Reactive S0 absorbs the queue on the critical path; proactive S1 hides it.
    assert abs((queued["T_S0"][0] - base["T_S0"][0]) - 50.0) < 1e-9
    assert abs(queued["T_S1"][0] - base["T_S1"][0]) < 1e-9


def test_prefill_lookup_table_interpolation_and_clamp():
    table = km.PrefillLookupTable([1024, 4096], [16, 64],
                                  np.array([[1.0, 2.0], [3.0, 4.0]]))
    # Exact grid points.
    assert abs(table(1024, 16)[0] - 1.0) < 1e-9
    assert abs(table(4096, 64)[0] - 4.0) < 1e-9
    # Geometric midpoint interpolates halfway in log space.
    assert abs(table(2048, 32)[0] - 2.5) < 1e-9
    # Clamp outside the grid.
    assert abs(table(10_000_000, 1)[0] - 3.0) < 1e-9
    # Zero suffix -> no prefill needed.
    assert table(1024, 0)[0] == 0.0


def test_roofline_prefill_monotone_in_prefix_and_suffix():
    models = km.load_models()
    profiles = km.parse_profiles(km.load_prefill_profiles())
    m = models["qwen3-235b-a22b"]
    p = profiles["b300_tp8_base"]
    short = km.prefill_ms_roofline(4096, 512, model=m, profile=p)
    long = km.prefill_ms_roofline(524288, 512, model=m, profile=p)
    more = km.prefill_ms_roofline(4096, 4096, model=m, profile=p)
    assert long > short  # prefix-length dependence, not a constant tokens/s
    assert more > short


def test_classification_and_estimated_append():
    # cached-like: prefix grew by ~output; resend-like: append contains output.
    labels = km.classify_output_assignment(
        prefix_delta=np.array([1000.0, 0.0, -50000.0]),
        next_append=np.array([50.0, 1200.0, 100.0]),
        prev_output=np.array([1000.0, 1000.0, 60000.0]),
    )
    assert list(labels) == ["output_cached_like", "output_resend_like", "ambiguous"]
    est = km.estimated_tool_append_tokens(
        labels, next_append=np.array([50.0, 1200.0, 100.0]),
        visible_output=np.array([900.0, 900.0, 900.0]))
    assert est[0] == 50.0            # cached-like: whole append is new input
    assert est[1] == 300.0           # resend-like: append minus visible output
    assert np.isnan(est[2])          # ambiguous -> excluded
