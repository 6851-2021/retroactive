"""Invariant-based tests for weight-balance B-trees."""
import pytest
from random import Random
from wb_btree import WBBTree

def random_seq(length, seed):
    rng = Random(seed)
    seq = list(range(length))
    rng.shuffle(seq)
    return seq

def make_sequences(prefix, length, num_random):
    return {
        **{
            f'{prefix}_asc': list(range(length)),
            f'{prefix}_desc': list(reversed(range(length)))
        }, **{
            f'{prefix}_rand_{seed}': random_seq(length, seed)
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

short_sequences = make_sequences('short', 500, 20)
long_sequences = make_sequences('long', 10000, 50)
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
        #all_iterator_invariant(tree, inserted)
        #find_invariant(tree, inserted)


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
            #all_iterator_invariant(tree, inserted)
            #find_invariant(tree, inserted)
    tree.check_invariants()
    #all_iterator_invariant(tree, inserted)
    #find_invariant(tree, inserted)
