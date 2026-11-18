"""Reservoir and coreset behaviour."""

import numpy as np

from models.memory import PatchReservoir, coreset_target_count, greedy_coreset


def test_reservoir_keeps_every_row_below_capacity():
    reservoir = PatchReservoir(capacity=10, seed=0)
    rows = np.arange(4, dtype=np.float32).reshape(4, 1)
    reservoir.add(rows)
    assert reservoir.seen == 4
    np.testing.assert_array_equal(reservoir.values().ravel(), np.arange(4))


def test_reservoir_never_exceeds_capacity():
    reservoir = PatchReservoir(capacity=8, seed=1)
    reservoir.add(np.arange(30, dtype=np.float32).reshape(30, 1))
    values = reservoir.values()
    assert values.shape == (8, 1)
    assert reservoir.seen == 30


def test_coreset_rows_come_from_the_input():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 5)).astype(np.float32)
    target = coreset_target_count(40, ratio=0.25, max_bank_size=100)
    selected = greedy_coreset(features, target, seed=0)
    assert selected.shape == (target, 5)
    original = {row.tobytes() for row in features}
    assert all(row.tobytes() in original for row in selected)


def test_coreset_target_respects_the_bank_cap():
    assert coreset_target_count(5000, 0.1, 2048) == 500
    assert coreset_target_count(3, 0.1, 2048) == 1
