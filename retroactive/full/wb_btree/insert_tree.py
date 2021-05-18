"""A range tree augmented to reflect inserts and deletions in another tree.

Used to maintain insertions masked by Qnow in a partially retroactive priority
queue.
"""
from typing import Generic, TypeVar, Optional, Tuple, Any, Callable
from range_scapegoat import RangeTree, RangeNode, RangeMeta, NodeType
from collections import deque
from functools import partial

TS = TypeVar('TS')
V = TypeVar('V')


def _cmp_or_none(cmp: Callable, a: Optional[Any],
                 b: Optional[Any]) -> Optional[Any]:
    if a is not None and b is not None:
        return cmp(a, b, key=lambda kv: kv[0])
    elif a is not None:
        return a
    elif b is not None:
        return b


_min_or_none = partial(_cmp_or_none, min)
_max_or_none = partial(_cmp_or_none, max)


class InsertMeta(RangeMeta):
    """Keeps track of metadata for min/max prefix sum lookups."""
    def __init__(self, *args, **kwargs):
        """Initializes range metadata."""
        self.max_absent = None
        self.min_present = None

    def insert(self, ts: TS, val: V, node: NodeType) -> None:
        """Updates range metadata to reflect an insert of a leaf."""
        self._propagate(node)

    def remove(self, ts: TS, val: V, node: NodeType) -> None:
        """Updates range metadata to reflect removal of a leaf."""
        self._propagate(node)

    def mark_present(self, val: V, node: NodeType) -> None:
        """Marks a value present (message from associated queue)."""
        if node.is_leaf and node.val == val:
            self.max_absent = None
            self.min_present = (val, node.min)
        else:
            self._propagate(node)

    def mark_absent(self, val: V, node: NodeType) -> None:
        """Marks a value absent (message from associated queue)."""
        if node.is_leaf and node.val == val:
            self.max_absent = (val, node.min)
            self.min_present = None
        else:
            self._propagate(node)

    def _propagate(self, parent):
        """Propagate changes upward."""
        if parent:
            self.max_absent = _max_or_none(parent.left.meta.max_absent,
                                           parent.right.meta.max_absent)
            self.min_present = _min_or_none(parent.left.meta.min_present,
                                            parent.right.meta.min_present)


def insert_tree_rebuild(root: NodeType) -> NodeType:
    """Rebuilds from the bottom up in O(n) time."""
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
            parent.meta = InsertMeta()
            parent.meta.max_absent = _max_or_none(left.meta.max_absent,
                                                  right.meta.max_absent)
            parent.meta.min_present = _min_or_none(left.meta.min_present,
                                                   right.meta.min_present)
            next_level.append(parent)
        else:
            next_level.append(left)
        if not leaves and len(next_level) > 1:
            leaves = next_level
            next_level = deque()
    return next_level.pop()


class InsertTree(RangeTree[TS, V]):
    """Stores insertions in a partially retroactive priority queue."""
    def __init__(self):
        super().__init__(rebuild_fn=insert_tree_rebuild, meta_cls=InsertMeta)

    def mark_present(self, ts: TS):
        path = list(self.root.path(ts))
        if path[-1].is_leaf:
            for node in reversed(path):
                node.meta.mark_present(path[-1].val, node)

    def mark_absent(self, ts: TS):
        path = list(self.root.path(ts))
        if path[-1].is_leaf:
            for node in reversed(path):
                node.meta.mark_absent(path[-1].val, node)

    def max_absent_in_range(self, lb: TS,
                            ub: TS) -> Tuple[Optional[V], Optional[TS]]:
        if not self.root:
            return None, None
        return max((node.meta.max_absent
                    for node in self.root.nodes_in_range(lb, ub)
                    if node.meta.max_absent is not None),
                   key=lambda kv: kv[0],
                   default=(None, None))

    def min_present_in_range(self, lb: TS,
                             ub: TS) -> Tuple[Optional[V], Optional[TS]]:
        if not self.root:
            return None, None
        return min((node.meta.min_present
                    for node in self.root.nodes_in_range(lb, ub)
                    if node.meta.min_present is not None),
                   key=lambda kv: kv[0],
                   default=(None, None))
