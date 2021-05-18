"""An (unoptimized) fully retroactive priority queue."""
from typing import TypeVar, Optional, List
from collections import deque
from range_scapegoat import RangeTree, RangeNode, RangeMeta, NodeType
from partial_pq import (PRPriorityQueue, EventType, INSERT, DELETE_MIN, TS,
                        TS_ZERO, TS_EPSILON)

V = TypeVar('V')


class PRPQMeta(RangeMeta):
    """Wrapper around a fully retroactive priority queue."""
    def __init__(self, *args, **kwargs):
        """Initializes an empty queue."""
        self.queue = PRPriorityQueue()
        if len(args) == 3:
            self.insert(*args)

    def insert(self, ts: TS, v: EventType, node: NodeType) -> None:
        """Updates the priority queue with a new event."""
        if v == DELETE_MIN:
            self.queue.delete_min(ts)
        else:
            self.queue.insert(ts, v[1])

    def remove(self, ts: TS, val: V, node: NodeType) -> None:
        """Removes an event from the priority queue."""
        self.queue.delete_op(ts)

    def __repr__(self):
        return f'priority queue with {self.queue.events.size} events'


def prpq_tree_rebuild(root: NodeType) -> NodeType:
    leaves = deque(leaf for _, leaf in root.leaves())
    next_level = deque()
    while leaves:
        left = leaves.popleft()
        if leaves:
            right = leaves.popleft()
            parent = RangeNode(left.min, None)
            parent.left = left
            parent.right = right
            parent.max = right.max
            parent.ub = left.ub + right.ub
            parent.size = left.size + right.size
            parent.meta = PRPQMeta()
            for t, event, _ in left.all():
                parent.meta.insert(t, event, parent)
            for t, event, _ in right.all():
                parent.meta.insert(t, event, parent)
            next_level.append(parent)
        else:
            next_level.append(left)
        if not leaves and len(next_level) > 1:
            leaves = next_level
            next_level = deque()
    return next_level.pop()


class PriorityQueue:
    """A fully retroactive priority queue."""
    def __init__(self):
        self.tree: RangeTree[TS, V] = RangeTree(rebuild_fn=prpq_tree_rebuild,
                                                meta_cls=PRPQMeta)
        self.t_next = 0

    def insert(self, val: V, t: Optional[TS] = None) -> None:
        """Inserts a value in the priority queue at time `t`.

        If `t` is not specified, the value is inserted strictly
        after the latest event time in the queue."""
        if t is None:
            self.t_next += TS_EPSILON
            t = self.t_next
        elif t <= TS_ZERO:
            raise ValueError(f'timestamp must be > {TS_ZERO}.')
        self.tree.insert(t, (INSERT, val))

    def delete_min(self, t: Optional[TS] = None) -> None:
        """Inserts a delete-min operation at time `t`.

        If the queue is empty at time `t`, a `ValueError` is raised.
        (The original retroactivity papers allow deleting from an
        empty queue; this is a slight simplifiation.)
        """
        if t is None:
            self.t_next += TS_EPSILON
            t = self.t_next
        elif t <= TS_ZERO:
            raise ValueError(f'timestamp must be > {TS_ZERO}.')
        self.tree.insert(t, DELETE_MIN)

    def delete_op(self, t: TS) -> None:
        """Deletes the operation at time `t` from the queue.

        If no event exists at time `t`, a `ValueError` is raised.
        """
        self.tree.remove(t)

    def at(self, t: TS) -> List[V]:
        """Fuse O(log m) queues to get the contents of the queue at `t`."""
        queues = list(self.tree.nodes_in_range(0, t))
        if not queues:
            return []
        q1_now = [v for v, _ in queues[0].meta.queue.now.all()]
        q1_deleted = [v for v, _ in queues[0].meta.queue.deleted.all()]
        for queue in queues[1:]:
            q2_now = [v for v, _ in queue.meta.queue.now.all()]
            q2_deleted = [v for v, _ in queue.meta.queue.deleted.all()]
            if len(q2_deleted) == 0:
                q3_now = q2_now + q1_now
                q3_deleted = q1_deleted
            elif len(q1_now) < len(q2_deleted):
                q3_now = q2_now
                q3_deleted = q1_deleted + q1_now + q2_deleted
            else:
                joined = sorted(q1_now + q2_deleted)
                split_idx = len(q1_now) - len(q2_deleted)
                q3_now = q2_now + joined[-split_idx:]
                q3_deleted = q1_deleted + joined[:len(q2_deleted)]
            q1_now = q3_now
            q1_deleted = q3_deleted
        return q1_now

    def __repr__(self):
        return '(root node) ' + str(self.tree.root.meta.queue)
