"""Unit tests for the retain-thinking transfer model."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kv_models as km


def test_kv_block_rounding():
    assert km.rounded_tokens(1, 16) == 16
    assert km.rounded_tokens(16, 16) == 16
    assert km.rounded_tokens(17, 16) == 32
    assert km.rounded_tokens(0, 16) == 0
    np.testing.assert_array_equal(
        km.rounded_tokens(np.array([1, 63, 64, 65]), 64),
        np.array([64.0, 64.0, 64.0, 128.0]),
    )


def test_transfer_bytes_multiplier():
    # 100 tokens at 1000 B/token, block 10 -> 100 rounded -> 100_000 B; x2 replication
    assert km.transfer_bytes(100, 1000, 10, 2.0) == 200_000


def test_network_unit_conversion():
    # 1e9 bytes at 1 GB/s == 1000 ms of wire time (+0 overhead).
    # 1e9 bytes = 1_000_000 tokens * 1000 B/token, block 1.
    t = km.transfer_ms(1_000_000, bytes_per_token=1000, block_size_tokens=1,
                       bandwidth_gbps=1.0, fixed_overhead_ms=0.0)
    assert abs(t - 1000.0) < 1e-9
    # 100 GB/s -> 10 ms; +0.5 ms overhead.
    t = km.transfer_ms(1_000_000, bytes_per_token=1000, block_size_tokens=1,
                       bandwidth_gbps=100.0, fixed_overhead_ms=0.5)
    assert abs(t - 10.5) < 1e-9


def test_overlap_split_hidden_exposed():
    hidden, exposed, full, frac = km.overlap_split(
        np.array([5.0, 20.0, 10.0]), np.array([10.0, 10.0, 10.0]))
    np.testing.assert_allclose(hidden, [5.0, 10.0, 10.0])
    np.testing.assert_allclose(exposed, [0.0, 10.0, 0.0])
    np.testing.assert_array_equal(full, [True, False, True])
    np.testing.assert_allclose(frac, [1.0, 0.5, 1.0])


def test_negative_slack_clips_to_zero():
    # tool ready *before* generation end -> slack must clip to 0, not negative.
    slack = km.clip_slack_ms(np.array([1_000_000]), np.array([2_000_000]))
    assert slack[0] == 0.0
    slack = km.clip_slack_ms(np.array([2_000_000]), np.array([1_000_000]))
    assert slack[0] == 1000.0  # 1e6 us = 1000 ms


def test_zero_transfer_full_overlap_fraction():
    _, _, full, frac = km.overlap_split(np.array([0.0]), np.array([0.0]))
    assert full[0]
    assert frac[0] == 1.0
