"""Tests for the prefix sum tree.

We particularly emphasize our use case: sequences with elements {-1, 0, 1}.
"""
import pytest
from random import Random
from collections import deque
from prefix_sum_tree import PrefixSumTree


def cumsum(x):
    """The cumulative sum of an array."""
    total = 0
    sums = []
    for el in x:
        total += el
        sums.append(total)
    return sums


def random_weight_seq(length, seed, lb=-1, ub=1):
    rng = Random(seed)
    return [rng.randint(lb, ub) for _ in range(length)]


def make_random_weight_sequences(prefix, length, num_random, seed_offset=0):
    return {
        f'{prefix}_rand_{seed}': random_weight_seq(length, seed_offset + seed)
        for seed in range(num_random)
    }


short_sequences = make_random_weight_sequences('short', 1000, 20)


@pytest.mark.parametrize('seq_name', short_sequences.keys())
def test_min_max_prefix_sum_at(seq_name):
    seq = short_sequences[seq_name]
    prefix_sums = cumsum(seq)
    min_prefix_sums = [min(0, prefix_sums[0])]
    max_prefix_sums = [max(prefix_sums[0], 0)]
    for idx in range(1, len(seq)):
        min_prefix_sums.append(min(min_prefix_sums[-1], prefix_sums[idx]))
        max_prefix_sums.append(max(max_prefix_sums[-1], prefix_sums[idx]))

    tree = PrefixSumTree(0)
    for idx, val in enumerate(seq):
        tree.insert(idx, val)
    for idx in range(len(seq)):
        min_prefix, max_prefix = tree.min_max_prefix_sum_at(idx)
        assert min_prefix == min_prefix_sums[idx]
        assert max_prefix == max_prefix_sums[idx]


@pytest.mark.parametrize('seq_name', short_sequences.keys())
def test_first_node_with_sum(seq_name):
    seq = short_sequences[seq_name]
    prefix_sums = cumsum(seq)
    zero_indices = deque(idx for idx, s in enumerate(prefix_sums) if s == 0)

    tree = PrefixSumTree(0)
    for idx, val in enumerate(seq):
        tree.insert(idx, val)
    for idx in range(len(seq)):
        first_node = tree.first_node_with_sum(idx, 0)
        if zero_indices and idx >= zero_indices[0]:
            zero_indices.popleft()
        if zero_indices:
            assert first_node.min == first_node.max == zero_indices[0]
        else:
            assert first_node is None


@pytest.mark.parametrize('seq_name', short_sequences.keys())
def test_last_node_with_sum(seq_name):
    seq = short_sequences[seq_name]
    prefix_sums = cumsum(seq)
    zero_indices = deque(idx for idx, s in enumerate(prefix_sums) if s == 0)

    tree = PrefixSumTree(0)
    for idx, val in enumerate(seq):
        tree.insert(idx, val)
    for idx in range(len(seq) - 1, -1, -1):
        last_node = tree.last_node_with_sum(idx, 0)
        if zero_indices and idx <= zero_indices[-1]:
            zero_indices.pop()
        if zero_indices:
            assert last_node.min == last_node.max == zero_indices[-1]
        else:
            assert last_node is None
