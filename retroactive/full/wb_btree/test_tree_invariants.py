"""Invariant tests for scapegoat range trees and weight-balanced B-trees."""
import pytest
from random import Random
from wb_btree import WBBTree
from range_scapegoat import SumRangeTree
from math import log2


def random_seq(length, seed):
    rng = Random(seed)
    seq = list(range(length))
    rng.shuffle(seq)
    return seq


def make_sequences(prefix, length, num_random, seed_offset=0):
    """Generates insertion (or deletion) sequences."""
    # The bit reversal sequence makes the most sense when `n`
    # is a power of 2, so we round.
    nearest_pow = round(log2(length))
    return {
        **{
            f'{prefix}_asc':
            list(range(length)),
            f'{prefix}_desc':
            list(reversed(range(length))),
            # elegant bit reversal: https://stackoverflow.com/a/12682003
            f'{prefix}_bit_reversal': [
                int(('{:0' + str(nearest_pow) + 'b}').format(n)[::-1], 2) for n in range(2**nearest_pow)
            ]
        },
        **{
            f'{prefix}_rand_{seed}': random_seq(length, seed + seed_offset)
            for seed in range(num_random)
        }
    }


def all_iterator_kv_invariant(tree, seq):
    """Invariant: all() generates key-value pairs in ascending order.

    (Used for `WBBTree` invariant checks.)
    """
    for key, (actual_k, actual_v) in zip(sorted(seq), tree.all()):
        assert key == actual_k, f"Keys don't match ({key} ≠ {actual_k})"
        assert key == actual_v, f"Values don't match ({key} ≠ {actual_v})"


def all_iterator_kv_meta_invariant(tree, seq):
    """Invariant: all() generates key-value-meta pairs in ascending order.

    (Used for `SumRangeTree` invariant checks.)
    """
    for key, (actual_k, actual_v, actual_meta) in zip(sorted(seq), tree.all()):
        assert key == actual_k, f"Keys don't match ({key} ≠ {actual_k})"
        assert key == actual_v, f"Values don't match ({key} ≠ {actual_v})"
        assert key == actual_meta.val, f"Sum wrong ({key} ≠ {actual_meta})"


def find_invariant(tree, seq):
    """Invariant: inserted keys can be found and return the right value."""
    for el in seq:
        assert tree.find(el) == el


def not_found_invariant(tree, seq):
    """Invariant: deleted keys cannot be found."""
    for el in seq:
        print('found', tree.find(el), 'for', el)
        assert tree.find(el) is None


b_tree_short_sequences = make_sequences('short', 500, 20)
b_tree_long_sequences = make_sequences('long', 10000, 50)
b_tree_insert_and_delete_sequences = {
    k: (ins_seq, del_seq)
    for ((k, ins_seq), (_, del_seq)) in zip(
        make_sequences('shortest', 250, 20).items(),
        make_sequences('shortest', 250, 20, seed_offset=20).items())
}
d_vals = [5, 8, 16, 32]

# For the same number of keys, scapegoat range trees have more nodes
# and fewer children per internal node than weight-balanced B-trees,
# so we use smaller test cases to achieve comparable test runtimes.
range_tree_short_sequences = make_sequences('short', 100, 20)
range_tree_long_sequences = make_sequences('long', 2000, 50)
range_tree_insert_and_delete_sequences = {
    k: (ins_seq, del_seq)
    for ((k, ins_seq), (_, del_seq)) in zip(
        make_sequences('shortest', 50, 20).items(),
        make_sequences('shortest', 50, 20, seed_offset=20).items())
}
alpha_vals = [0.51, 0.67, 0.99]


# Strict tests check invariants at every step.
@pytest.mark.parametrize('seq', b_tree_short_sequences.keys())
@pytest.mark.parametrize('d', d_vals)
def test_b_tree_insert_only_strict(seq, d):
    tree = WBBTree(d=d)
    tree.check_invariants()
    inserted = []
    for el in b_tree_short_sequences[seq]:
        tree.insert(el, el)
        inserted.append(el)
        tree.check_invariants()
        all_iterator_kv_invariant(tree, inserted)
        find_invariant(tree, inserted)


@pytest.mark.parametrize('seq', range_tree_short_sequences.keys())
@pytest.mark.parametrize('alpha', alpha_vals)
def test_sum_range_tree_insert_only_strict(seq, alpha):
    tree = SumRangeTree(alpha=alpha)
    tree.check_invariants()
    inserted = []
    for el in range_tree_short_sequences[seq]:
        print(el)
        tree.insert(el, el)
        inserted.append(el)
        tree.check_invariants()
        all_iterator_kv_meta_invariant(tree, inserted)
        find_invariant(tree, inserted)


@pytest.mark.parametrize('seq', b_tree_long_sequences.keys())
@pytest.mark.parametrize('d', d_vals)
def test_b_tree_insert_only(seq, d):
    tree = WBBTree(d=d)
    tree.check_invariants()
    inserted = []
    for idx, el in enumerate(b_tree_long_sequences[seq]):
        tree.insert(el, el)
        inserted.append(el)
        if idx % 2000 == 0:
            tree.check_invariants()
            all_iterator_kv_invariant(tree, inserted)
            find_invariant(tree, inserted)
    tree.check_invariants()
    all_iterator_kv_invariant(tree, inserted)
    find_invariant(tree, inserted)


@pytest.mark.parametrize('seq', range_tree_long_sequences.keys())
@pytest.mark.parametrize('alpha', alpha_vals)
def test_sum_range_tree_insert_only(seq, alpha):
    tree = SumRangeTree(alpha=alpha)
    tree.check_invariants()
    inserted = []
    for idx, el in enumerate(range_tree_long_sequences[seq]):
        tree.insert(el, el)
        inserted.append(el)
        if idx % 2000 == 0:
            tree.check_invariants()
            all_iterator_kv_meta_invariant(tree, inserted)
            find_invariant(tree, inserted)
    tree.check_invariants()
    all_iterator_kv_meta_invariant(tree, inserted)
    find_invariant(tree, inserted)


@pytest.mark.parametrize('seq_pair', b_tree_insert_and_delete_sequences.keys())
@pytest.mark.parametrize('d', d_vals)
def test_b_tree_insert_and_delete_strict(seq_pair, d):
    insert_seq, delete_seq = b_tree_insert_and_delete_sequences[seq_pair]
    tree = WBBTree(d=d)
    tree.check_invariants()
    inserted = []
    for el in insert_seq:
        tree.insert(el, el)
        inserted.append(el)
        tree.check_invariants()
        all_iterator_kv_invariant(tree, inserted)
        find_invariant(tree, inserted)
    deleted = []
    for el in delete_seq:
        tree.remove(el)
        inserted.remove(el)
        deleted.append(el)
        tree.check_invariants()
        all_iterator_kv_invariant(tree, inserted)
        find_invariant(tree, inserted)
        not_found_invariant(tree, deleted)


@pytest.mark.parametrize('seq_pair',
                         range_tree_insert_and_delete_sequences.keys())
@pytest.mark.parametrize('alpha', alpha_vals)
def test_sum_range_tree_insert_and_delete_strict(seq_pair, alpha):
    insert_seq, delete_seq = range_tree_insert_and_delete_sequences[seq_pair]
    tree = SumRangeTree(alpha=alpha)
    tree.check_invariants()
    inserted = []
    for el in insert_seq:
        tree.insert(el, el)
        inserted.append(el)
        tree.check_invariants()
        all_iterator_kv_meta_invariant(tree, inserted)
        find_invariant(tree, inserted)
    deleted = []
    for el in delete_seq:
        tree.remove(el)
        inserted.remove(el)
        deleted.append(el)
        tree.check_invariants()
        all_iterator_kv_meta_invariant(tree, inserted)
        find_invariant(tree, inserted)
        not_found_invariant(tree, deleted)


@pytest.mark.parametrize('seq', range_tree_long_sequences.keys())
def test_sum_range_tree_insert_nodes_in_range(seq):
    tree = SumRangeTree()
    for el in range_tree_long_sequences[seq]:
        tree.insert(el, el)
    keys = []
    for key, val, meta in tree.in_range(100, 500):
        assert key == val == meta.val
        keys.append(key)
    assert keys == list(range(100, 501))
