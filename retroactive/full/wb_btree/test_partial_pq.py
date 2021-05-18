"""Basic tests for the partially retroactive priority queue."""
from partial_pq import PRPriorityQueue

def test_partial_pq_insert_only():
    pq = PRPriorityQueue()
    for i in range(1, 101):
        pq.insert(i)
    assert list(pq.all()) == list(range(1, 101))


def test_partial_pq_insert_then_delete_min():
    pq = PRPriorityQueue()
    for i in range(1, 101):
        pq.insert(i)
    pq.delete_min()
    pq.delete_min()
    assert list(pq.all()) == list(range(3, 101))


def test_partial_pq_delete_delete_min():
    pq = PRPriorityQueue()
    for i in range(1, 101):
        pq.insert(i)
    pq.delete_min(t=101)
    pq.delete_min(t=102)
    pq.delete_op(101)
    pq.delete_op(102)
    assert list(pq.all()) == list(range(1, 101))


def test_partial_pq_delete_insert():
    pq = PRPriorityQueue()
    for i in range(1, 101):
        pq.insert(i)
    pq.delete_op(1)
    pq.delete_op(2)
    assert list(pq.all()) == list(range(3, 101))
