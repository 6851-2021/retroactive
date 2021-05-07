"""Invariant-based tests for weight-balance B-trees."""
import pytest
from random import Random
from wb_btree import WBBTree

def random_seq(length, seed):
    rng = Random(seed)
    seq = list(range(length))
    rng.shuffle(seq)
    return seq

def make_sequences(prefix, length, num_random, seed_offset=0):
    return {
        **{
            f'{prefix}_asc': list(range(length)),
            f'{prefix}_desc': list(reversed(range(length))),
        }, **{
            f'{prefix}_rand_{seed}': random_seq(length, seed + seed_offset)
            for seed in range(num_random)
        }
    }

def all_iterator_invariant(tree, seq):
    """Invariant: all() generates key-value pairs in ascending order."""
    for key, (actual_k, actual_v) in zip(sorted(seq), tree.all()):
        assert key == actual_k, f"Keys don't match ({key} ≠ {actual_k})"
        assert key == actual_v, f"Values don't match ({key} ≠ {actual_v})"

def find_invariant(tree, seq):
    """Invariant: inserted keys can be found and return the right value."""
    for el in seq:
        assert tree.find(el) == el

def not_found_invariant(tree, seq):
    """Invariant: deleted keys cannot be found."""
    for el in seq:
        print('found', tree.find(el), 'for', el)
        assert tree.find(el) is None

short_sequences = make_sequences('short', 500, 20)
long_sequences = make_sequences('long', 10000, 50)
insert_and_delete_sequences = {
    k: (ins_seq, del_seq)
    for ((k, ins_seq), (_, del_seq))
    in zip(make_sequences('shortest', 250, 20).items(),
           make_sequences('shortest', 250, 20, seed_offset=20).items())
}
d_vals = [5, 8, 16, 32]

# Strict tests check invariants at every step.
@pytest.mark.parametrize('seq', short_sequences.keys())
@pytest.mark.parametrize('d', d_vals)
def test_insert_only_strict(seq, d):
    tree = WBBTree(d=d)
    tree.check_invariants()
    inserted = []
    for el in short_sequences[seq]:
        print(el)
        tree.insert(el, el)
        inserted.append(el)
        tree.check_invariants()
        all_iterator_invariant(tree, inserted)
        find_invariant(tree, inserted)


@pytest.mark.parametrize('seq', long_sequences.keys())
@pytest.mark.parametrize('d', d_vals)
def test_insert_only(seq, d):
    tree = WBBTree(d=d)
    tree.check_invariants()
    inserted = []
    for idx, el in enumerate(long_sequences[seq]):
        tree.insert(el, el)
        inserted.append(el)
        if idx % 2000 == 0:
            tree.check_invariants()
            all_iterator_invariant(tree, inserted)
            find_invariant(tree, inserted)
    tree.check_invariants()
    all_iterator_invariant(tree, inserted)
    find_invariant(tree, inserted)


@pytest.mark.parametrize('seq_pair', insert_and_delete_sequences.keys())
@pytest.mark.parametrize('d', d_vals)
def test_insert_and_delete_strict(seq_pair, d):
    insert_seq, delete_seq = insert_and_delete_sequences[seq_pair]
    tree = WBBTree(d=d)
    tree.check_invariants()
    inserted = []
    for el in insert_seq:
        tree.insert(el, el)
        inserted.append(el)
        tree.check_invariants()
        all_iterator_invariant(tree, inserted)
        find_invariant(tree, inserted)
    deleted = []
    for el in delete_seq:
        tree.delete(el)
        inserted.remove(el)
        deleted.append(el)
        tree.check_invariants()
        all_iterator_invariant(tree, inserted)
        find_invariant(tree, inserted)
        not_found_invariant(tree, deleted)
