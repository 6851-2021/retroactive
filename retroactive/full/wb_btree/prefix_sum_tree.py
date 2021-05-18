from typing import Generic, TypeVar, Optional, Tuple, Any
from range_scapegoat import RangeTree, RangeNode, RangeMeta, NodeType
from collections import deque

K = TypeVar('K')
V = TypeVar('V')


def make_meta(zero: Any):
    """Generates a metadata class and rebuild function with a typed zero."""
    class PrefixSumMeta(RangeMeta):
        """Keeps track of metadata for min/max prefix sum lookups."""
        def __init__(self, key: K, val: V, node: NodeType):
            """Initializes range metadata."""
            self.sum = val
            self.min_prefix_sum = val
            self.max_prefix_sum = val

        def insert(self, key: K, val: V, node: NodeType) -> None:
            """Updates range metadata to reflect an insert of a leaf."""
            self.sum += val
            self._propagate(node)

        def remove(self, key: K, val: V, node: NodeType) -> None:
            """Updates range metadata to reflect removal of a leaf."""
            self.sum -= val
            self._propagate(node)

        def _propagate(self, parent: NodeType) -> None:
            """Propagate changes upward."""
            if parent:
                self.min_prefix_sum = min(
                    parent.left.meta.min_prefix_sum,
                    parent.left.meta.sum + parent.right.meta.min_prefix_sum)
                parent.meta.max_prefix_sum = max(
                    parent.left.meta.max_prefix_sum,
                    parent.left.meta.sum + parent.right.meta.max_prefix_sum)

        def __repr__(self):
            rep = f'sum: {self.sum}'
            if self.min_prefix_sum is not None:
                rep += f', subtree min prefix sum: {self.min_prefix_sum}'
            if self.max_prefix_sum is not None:
                rep += f', subtree max prefix sum: {self.max_prefix_sum}'
            return rep

    def prefix_sum_rebuild(root: NodeType) -> NodeType:
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
                parent.meta = PrefixSumMeta(left.min, left.meta.sum, left)
                parent.meta.sum = left.meta.sum + right.meta.sum
                parent.meta.min_prefix_sum = min(
                    left.meta.min_prefix_sum,
                    left.meta.sum + right.meta.min_prefix_sum)
                parent.meta.max_prefix_sum = max(
                    left.meta.max_prefix_sum,
                    left.meta.sum + right.meta.max_prefix_sum)
                next_level.append(parent)
            else:
                next_level.append(left)
            if not leaves and len(next_level) > 1:
                leaves = next_level
                next_level = deque()
        return next_level.pop()

    return PrefixSumMeta, prefix_sum_rebuild


class PrefixSumTree(RangeTree[K, V]):
    """A tree that supports prefix sum queries."""
    def __init__(self, zero: Optional[V] = None):
        PrefixSumMeta, prefix_rebuild = make_meta(zero)
        super().__init__(rebuild_fn=prefix_rebuild, meta_cls=PrefixSumMeta)
        self.zero = zero

    def min_max_prefix_sum_at(self, key: K) -> Optional[Tuple[V, V]]:
        """Finds the prefix sums at a node."""
        if not self.root:
            return None
        path = list(self.root.path(key))
        leaf = path[-1]
        if leaf.is_leaf and leaf.min == key:
            path_sum = self.zero
            min_sum = self.zero
            max_sum = self.zero
            for (parent, child) in zip(path[:-1], path[1:]):
                if child == parent.right:
                    # Right turn.
                    min_sum = min(min_sum,
                                  path_sum + parent.left.meta.min_prefix_sum)
                    max_sum = max(max_sum,
                                  path_sum + parent.left.meta.max_prefix_sum)
                    path_sum += parent.left.meta.sum
            min_sum = min(min_sum, path_sum + leaf.meta.min_prefix_sum)
            max_sum = max(max_sum, path_sum + leaf.meta.max_prefix_sum)
            return min_sum, max_sum

    def min_prefix_sum_at(self, key: K) -> Optional[V]:
        """Finds the minimum prefix sum at a node."""
        min_max = self.min_max_prefix_sum_at(key)
        if min_max:
            return min_max[0]

    def max_prefix_sum_at(self, key: K) -> Optional[V]:
        """Finds the maximum prefix sum at a node."""
        min_max = self.min_max_prefix_sum_at(key)
        if min_max:
            return min_max[1]

    def last_node_with_sum(self, key_ub: K,
                           prefix_sum: V) -> Optional[NodeType]:
        """Finds the last node such that:
            * The key is < key_ub.
            * The prefix sum at the tree is exactly `prefix_sum`.

        If such a node cannot be found, `None` is returned."""
        if not self.root:
            return None
        path = list(self.root.path_left_biased(key_ub))
        if path[-1].min == key_ub:
            pred = self.root.predecessor(key_ub)
            if pred is None:
                return None
            path = list(self.root.path_left_biased(pred))
        path_sum = self.zero
        for parent, child in zip(path[:-1], path[1:]):
            if child == parent.right:
                path_sum += parent.left.meta.sum

        # Hopefully, we got lucky and found a node with the right prefix sum.
        # Otherwise, we work backward.
        if prefix_sum == path_sum + path[-1].meta.sum:
            return path[-1]

        node = None
        for parent, child in zip(reversed(path[:-1]), reversed(path[1:])):
            left_lb = (path_sum - parent.left.meta.sum +
                       parent.left.meta.min_prefix_sum)
            left_ub = (path_sum - parent.left.meta.sum +
                       parent.left.meta.max_prefix_sum)
            if child == parent.right:
                path_sum -= parent.left.meta.sum
                # Move back up the tree until we find the first unexplored left
                # subtree with the prefix sum in bounds.
                if left_lb <= prefix_sum <= left_ub:
                    node = parent.left
                    break

        # Walk to the rightmost possible node in the subtree we identified.
        while node and not node.is_leaf:
            left_sum = node.left.meta.sum
            left_lb = node.left.meta.min_prefix_sum + path_sum
            left_ub = node.left.meta.max_prefix_sum + path_sum
            right_lb = node.right.meta.min_prefix_sum + path_sum + left_sum
            right_ub = node.right.meta.max_prefix_sum + path_sum + left_sum
            if node.right.min <= key_ub and \
               right_lb <= prefix_sum <= right_ub:
                node = node.right
                path_sum += left_sum
            elif left_lb <= prefix_sum <= left_ub:
                node = node.left
            else:
                break
        if node and node.is_leaf:
            return node

    def first_node_with_sum(self, key_lb: K,
                            prefix_sum: V) -> Optional[NodeType]:
        """Finds the last node such that:
            * The key is > key_lb.
            * The prefix sum at the tree is exactly `prefix_sum`.

        If such a node cannot be found, `None` is returned."""
        if not self.root:
            return None
        path = list(self.root.path(key_lb))
        if path[-1].min == key_lb:
            succ = self.root.successor(key_lb)
            if succ is None:
                return None
            path = list(self.root.path(succ))
        path_sum = self.zero
        for parent, child in zip(path[:-1], path[1:]):
            if child == parent.right:
                path_sum += parent.left.meta.sum

        # Hopefully, we got lucky and found a node with the right prefix sum.
        # Otherwise, we work backward.
        if prefix_sum == path_sum + path[-1].meta.sum:
            return path[-1]

        node = None
        for parent, child in zip(reversed(path[:-1]), reversed(path[1:])):
            left_sum = parent.left.meta.sum
            right_lb = path_sum + left_sum + parent.right.meta.min_prefix_sum
            right_ub = path_sum + left_sum + parent.right.meta.max_prefix_sum
            if child == parent.left and right_lb <= prefix_sum <= right_ub:
                node = parent.right
                path_sum += parent.left.meta.sum
                break
            if child == parent.right:
                path_sum -= parent.left.meta.sum

        # Go leftward as much as possible, going left when left.max < key_lb
        # or the prefix sum constraint is violated.
        while node and not node.is_leaf:
            left_sum = node.left.meta.sum
            left_lb = node.left.meta.min_prefix_sum + path_sum
            left_ub = node.left.meta.max_prefix_sum + path_sum
            right_lb = node.right.meta.min_prefix_sum + path_sum + left_sum
            right_ub = node.right.meta.max_prefix_sum + path_sum + left_sum
            if node.left.max >= key_lb and left_lb <= prefix_sum <= left_ub:
                node = node.left
            elif right_lb <= prefix_sum <= right_ub:
                node = node.right
                path_sum += left_sum
            else:
                break
        if node and node.is_leaf:
            return node
