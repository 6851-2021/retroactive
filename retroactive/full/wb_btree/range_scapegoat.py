"""A dynamic range tree based on the scapegoat tree."""
from typing import (TypeVar, Generic, Optional, List, Tuple, Generator,
                    Callable, Iterable, Any)
from abc import ABC, abstractmethod
from copy import copy
from math import log, floor
from collections import deque

K = TypeVar('K')
V = TypeVar('V')
MetaType = Optional[Any]  # TODO: improve
KVMeta = Tuple[K, V, MetaType]
NodeType = 'RangeNode[K, V, Meta]'
TreeType = 'RangeTree[K, V]'
RebuildType = Callable[[NodeType], NodeType]
NodeIterator = Generator[NodeType, None, None]
KVMetaIterator = Generator[KVMeta, None, None]

# partially based on
# https://opendatastructures.org/newhtml/ods/latex/scapegoat.html
# TODO: full citations

class RangeMeta(ABC):
    """Abstract base class for range metadata."""
    @abstractmethod
    def __init__(self, key: K, val: V, node: NodeType):
        """Initializes range metadata."""

    @abstractmethod
    def insert(self, key: K, val: V, node: NodeType) -> None:
        """Updates range metadata to reflect an insert of a leaf."""

    @abstractmethod
    def remove(self, key: K, val: V, node: NodeType) -> None:
        """Updates range metadata to reflect removal of a leaf."""


class RangeTree(Generic[K, V]):
    """A dynamic range tree based on the scapegoat tree."""
    def __init__(self,
                 rebuild_fn: RebuildType,
                 meta_cls: MetaType = None,
                 alpha: float = 2 / 3):
        """Creates a dynamic range tree with scapegoat rebalancing.

        Each internal node represents a range; leaves represent key-value
        pairs. Every node can have metadata attached.

        This is intended as a base class; the metadata class (which implements
        the `RangeMeta` abstract base class) and rebuilding function must be
        provided by the user, as there are applications where rebuilding
        can be done bottom-up in O(n) time (e.g. trees with simple aggregations
        like summing) and applications where rebuilding is slower and must be
        done in a custom way (e.g. trees of fusible partially retroactive
        priority queues).

        Args:
            rebuild_fn: A function that takes a subtree rooted at a given
              `RangeNode` and rebuilds the tree to be (approximately) balanced
              with metadata preserved.
            meta_cls: The node-level metadata class, which must implement
              `RangeMeta`.
            alpha: The balance parameter (0.5 < alpha < 1). Lower values will
              result in a more strictly balanced tree, but more rebuilds
              are required in the worst case (slow inserts, fast searches);
              higher values result in a loosely balanced tree that is not
              rebuilt as frequently (fast inserts, slow searches).
        """
        if alpha < 0.5:
            raise ValueError('α must be at least 0.5.')
        if alpha > 1:
            raise ValueError('α must be less than 1.')
        self._meta_cls = meta_cls
        self._rebuild_fn = rebuild_fn
        self.alpha = alpha
        self.root: Optional[NodeType] = None

    def find(self, key: K) -> Optional[V]:
        """Searches for a key in the tree.

        If the key is found, its associate value is returned. Otherwise,
        `None` is returned."""
        if self.root:
            return self.root.find(key)

    def insert(self, key: K, val: V) -> None:
        """Inserts a key-value pair into the tree.

        If `key` is already in the tree, a `ValueError`` is raised."""
        if self.root:
            self.root = self.root.insert(key, val, self._rebuild_fn,
                                         self.alpha)
        else:
            self.root = RangeNode(key, val, self._meta_cls)

    def remove(self, key: K) -> None:
        """Removes a key-value pair from the tree.

        If the `key` is not in the tree, a ``ValueError`` is raised."""
        if self.root:
            self.root = self.root.remove(key, self._rebuild_fn)
        else:
            raise ValueError('Cannot remove from an empty tree.')

    def all(self) -> KVMetaIterator:
        """Returns all key-value-meta tuples in order."""
        if self.root:
            yield from self.root.all()

    def predecessor(self, key: K) -> Optional[K]:
        """Finds the key immediately before `key`, if it exists."""
        if self.root:
            return self.root.predecessor(key)

    def successor(self, key: K) -> Optional[K]:
        """Finds the key immediately after `key`, if it exists."""
        if self.root:
            return self.root.successor(key)

    def in_range(self, lb: K, ub: K) -> KVMetaIterator:
        """Returns all key-value-meta tuples with keys in range [lb, ub]."""
        for node in self.nodes_in_range(lb, ub):
            yield from node.all()

    def nodes_in_range(self, lb: K, ub: K) -> NodeIterator:
        """Generates a minimal set of nodes (expected size O(log n)) covering
        the range [lb, ub]."""
        if self.root:
            return self.root.nodes_in_range(lb, ub)

    def _range_invariant(self) -> bool:
        """Invariant: an internal node's range is [left.min, right.max]."""
        return all(node.min == node.left.min and node.max == node.right.max
                   for node in self.root.internal_nodes())

    def _internal_order_invariant(self) -> bool:
        """Invariant: at each internal node,
        left.min ≤ left.max < right.min ≤ right.max."""
        return all(
            node.left.min <= node.left.max < node.right.min <= node.right.max
            for node in self.root.internal_nodes())

    def _internal_val_invariant(self) -> bool:
        """Invariant: internal nodes do not have values attached."""
        return all(node.val is None for node in self.root.internal_nodes())

    def _ub_size_invariant(self) -> bool:
        """Invariant: at the root node, ub ≤ 2 * size."""
        return self.root.ub <= 2 * self.root.size

    def _meta_invariant(self) -> bool:
        """Invariant: either
          (a) No nodes have metadata -or-
          (b) All nodes have metadata, and each metadata object is distinct."""
        if self._meta_cls:
            metas = [node.meta for node in self.root.all_nodes()]
            return (all(m is not None for m in metas)
                    and len(metas) == len(set(metas)))
        return all(node.meta is None for node in self.root.all_nodes())

    def _balance_invariant(self) -> bool:
        """Invariant: the tree is approximately balanced."""
        max_depth = max(depth for depth, _ in self.root.leaves())
        # see https://en.wikipedia.org/wiki/Scapegoat_tree for
        # ɑ-height-balance and ɑ-weight-balance invariants, which we
        # loosen slightly o account for internal nodes.
        depth_ub = floor(log(2 * (self.root.ub + 1)) / log(1 / self.alpha)) + 1
        print('max depth:', max_depth, '\tdepth ub:', depth_ub, '\tsize ub:', self.root.ub)
        return max_depth <= depth_ub

    def check_invariants(self) -> None:
        """Verifies that the tree is well-formed."""
        if self.root:
            assert self._range_invariant(), \
                "An internal node's range is not [left.min, right.max]."
            assert self._internal_order_invariant(), \
                ("An internal node's child ranges are out of order " +
                 "(expected: L.min ≤ L.max < R.min ≤ R.max).")
            assert self._internal_val_invariant(), \
                'An internal node has a value attached.'
            assert self._ub_size_invariant(), \
                'A node has too many deleted keys (ub > 2 * size).'
            assert self._meta_invariant(), 'A node has invalid metadata.'
            assert self._balance_invariant(), 'The tree is too unbalanced'


class RangeNode(Generic[K, V]):
    """A node in a dynamic range tree."""
    def __init__(self, key: K, val: Optional[V], meta_cls: MetaType = None):
        """Creates a new range node."""
        self.min = key
        self.max = key
        self.val = val
        if meta_cls:
            self.meta = meta_cls(key, val, self)
        else:
            self.meta = None
        self.left = None
        self.right = None
        self.size = 1
        self.ub = 1

    @property
    def is_leaf(self) -> bool:
        """Does the node have any children?"""
        return self.min == self.max

    def find(self, key: K) -> Optional[V]:
        """Looks up a key in the subtree rooted at the node.

        Returns:
            The value associated with the key, if a match is found.
            Otherwise, returns `None`.
        """
        node = None
        for node in self.path(key):
            pass  # Find the last node on the path.
        if node.is_leaf and node.min == key:
            return node.val

    def path(self, key: K) -> NodeIterator:
        """Finds a (right-biased) path to a (possibly nonexistent) key."""
        yield self
        if not self.is_leaf:
            if self.left and self.left.max >= key:
                yield from self.left.path(key)
            else:
                yield from self.right.path(key)

    def path_left_biased(self, key: K) -> NodeIterator:
        """Finds a (left-biased) path to a (possibly nonexistent) key."""
        yield self
        if not self.is_leaf:
            if self.right and self.right.min <= key:
                yield from self.right.path_left_biased(key)
            else:
                yield from self.left.path_left_biased(key)

    def insert(self, key: K, val: V, rebuild_fn: RebuildType,
               alpha: float) -> Tuple[NodeType, int]:
        """Inserts a key-value pair in the subtree rooted at the node,
        rebalancing if necessary (as determined by the balance factor ɑ)
        using `rebuild_fn`."""
        path = list(self.path(key))
        leaf = path[-1]
        if leaf.is_leaf and leaf.min == key:
            raise ValueError(f'Key "{key}" already in tree')
        if self.meta:
            meta_cls = self.meta.__class__
        else:
            meta_cls = None
        new_leaf = RangeNode(key, val, meta_cls)
        replacement_leaf = copy(leaf)
        replacement_leaf.meta = copy(leaf.meta)
        leaf.val = None
        if leaf.min > key:
            leaf.min = key
            leaf.left = new_leaf
            leaf.right = replacement_leaf
        else:
            leaf.max = key
            leaf.left = replacement_leaf
            leaf.right = new_leaf

        # Update metadata.
        if meta_cls:
            new_leaf.meta = meta_cls(key, val, new_leaf)
        for node in reversed(path):
            node.size += 1
            node.ub += 1
            node.min = min(node.min, key)
            node.max = max(node.max, key)
            if meta_cls:
                node.meta.insert(key, val, node)

        # Scapegoat rebuild.
        if len(path) > log(2 * self.ub) / log(1 / alpha):
            scapegoat = None
            parent = None
            for parent, child in zip(reversed(path[:-1]), reversed(path[1:])):
                if scapegoat:
                    break
                if child.size > int(alpha * parent.size) + 1:
                    scapegoat = parent
            if scapegoat is None or scapegoat == self:
                rebuilt = rebuild_fn(self)
                return rebuilt
            if scapegoat == parent.left:
                parent.left = rebuild_fn(scapegoat)
            else:
                parent.right = rebuild_fn(scapegoat)
        return self

    def remove(self, key: K, rebuild_fn: RebuildType) -> Optional[NodeType]:
        """Removes a key (if it exists) from the subtree rooted at the node.

        If the key does not exist in the subtree, a `ValueError` is raised.

        Returns:
            A new root, possibly resulting from a rebalancing with
            `rebuild_fn`.  If deleting the key yields an empty subtree,
            `None` is returned.
        """
        path = list(self.path(key))
        leaf = path[-1]
        if key != leaf.min or key != leaf.max:
            raise ValueError(f'Cannot delete "{key}" (not in tree)')
        if leaf == self:
            # 1 element in tree: the tree is now empty.
            return None
        parent = path[-2]
        if parent.left == leaf:
            sibling = parent.right
        else:
            sibling = parent.left
        if len(path) == 2:
            # Promote a depth-1 subtree to the root.
            if sibling.ub > 2 * sibling.size:
                return rebuild_fn(sibling)
            return sibling
        # ≥3 elements in tree: replace the leaf's parent with the leaf.
        grandparent = path[-3]
        if grandparent.left == parent:
            grandparent.left = sibling
        else:
            grandparent.right = sibling

        # Update metadata, etc.
        for node in reversed(path[:-2]):
            node.size -= 1
            node.min = node.left.min
            node.max = node.right.max
            if node.meta:
                node.meta.remove(key, leaf.val, node)

        # Lazy rebuilding.
        if self.ub > 2 * self.size:
            return rebuild_fn(self)
        return self

    def leaves(self) -> Generator[Tuple[int, NodeType], None, None]:
        """Finds all the leaves in the subtree rooted at the node."""
        if self.is_leaf:
            yield (0, self)
        else:
            yield from ((level + 1, node)
                        for level, node in self.left.leaves())
            yield from ((level + 1, node)
                        for level, node in self.right.leaves())

    def internal_nodes(self) -> NodeIterator:
        """Finds all the internal nodes in the subtree rooted at the node."""
        if not self.is_leaf:
            yield self
            yield from self.left.internal_nodes()
            yield from self.right.internal_nodes()

    def all_nodes(self) -> NodeIterator:
        """Finds all the nodes in the subtree rooted at the node."""
        yield self
        if not self.is_leaf:
            yield from self.left.all_nodes()
            yield from self.right.all_nodes()

    def all(self) -> KVMetaIterator:
        """Finds all the key-value pairs in the subtree rooted at the node."""
        if self.is_leaf:
            yield (self.min, self.val, self.meta)
        else:
            # In-order traversal.
            yield from self.left.all()
            yield from self.right.all()

    def predecessor(self, key: K) -> Optional[K]:
        """Finds the key immediately before `key`, if it exists."""
        path = list(self.path(key))
        leaf = path[-1]
        if not leaf.is_leaf:
            raise ValueError(f'Cannot find predecessor of missing key {key}.')
        for parent, child in zip(reversed(path[:-1]), reversed(path[1:])):
            if child == parent.right:
                curr = parent.left
                while not curr.is_leaf:
                    curr = curr.right
                return curr.min

    def successor(self, key: K) -> Optional[K]:
        """Finds the key immediately after `key`, if it exists."""
        path = list(self.path(key))
        leaf = path[-1]
        if not leaf.is_leaf:
            raise ValueError(f'Cannot find successor of missing key {key}.')
        for parent, child in zip(reversed(path[:-1]), reversed(path[1:])):
            if child == parent.left:
                curr = parent.right
                while not curr.is_leaf:
                    curr = curr.left
                return curr.min

    def nodes_in_range(self, lb: K, ub: K) -> NodeIterator:
        """Generates a minimal set of nodes (expected size O(log n)) covering
        the range [lb, ub]."""
        if lb > ub:
            raise ValueError('Invalid range query: ' +
                             f'lower bound ({lb}) > upper bound ({ub}).')
        if lb <= self.min <= self.max <= ub:
            # If the current node's range is completely contained in [lb, ub],
            # don't traverse any deeper.
            yield self
            return
        if self.left.max >= lb:
            yield from self.left.nodes_in_range(lb, ub)
        if self.right.min <= ub:
            yield from self.right.nodes_in_range(lb, ub)

    def __repr__(self):
        if self.is_leaf:
            desc = f'leaf with key {self.min}'
        else:
            desc = (f'internal node with key range [{self.min}, {self.max}] ' +
                    f'(size {self.size})')
        if self.meta:
            desc += f' (meta: {self.meta})'
        return desc


AggType = Callable[[V, V], V]


def make_agg_meta(insert_fn: AggType, remove_fn: AggType):
    """Generates a metadata class and rebuild function for a simple
    (i.e. commutative, etc.) aggregation (e.g. +, *)."""
    class AggMeta(RangeMeta):
        """Metadata class for an aggregation."""
        def __init__(self, key: K, val: V, *args, **kwargs):
            super().__init__(key, val, *args, **kwargs)
            self.val = val

        def insert(self, key: K, val: V, *args, **kwargs) -> None:
            self.val = insert_fn(self.val, val)

        def remove(self, key: K, val: V, *args, **kwargs) -> None:
            self.val = remove_fn(self.val, val)

        def __repr__(self):
            return str(self.val)

    def rebuild_fn(root: NodeType) -> NodeType:
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
                parent.ub = parent.left.ub + parent.right.ub
                parent.size = parent.left.size + parent.right.size
                parent.meta = AggMeta(left.min, left.meta.val)
                parent.meta.insert(right.max, right.meta.val)
                next_level.append(parent)
            else:
                next_level.append(left)
            if not leaves and len(next_level) > 1:
                leaves = next_level
                next_level = deque()
        return next_level.pop()

    class AggRangeTree(RangeTree):
        """A range tree with an aggregation and bottom-up rebalancing."""
        def __init__(self, *args, **kwargs):
            super().__init__(rebuild_fn, AggMeta, *args, **kwargs)

        def _agg_invariant(self) -> bool:
            """Invariant: the aggregation is consistent."""
            return all(node.meta.val ==
                       insert_fn(node.left.meta.val, node.right.meta.val)
                       for node in self.root.internal_nodes())

        def check_invariants(self):
            """Checks all tree invariants, plus the aggregation invariant."""
            super().check_invariants()
            if self.root:
                assert self._agg_invariant(), \
                    ("A node's aggregation metadata is inconsistent with " +
                     "its children's.")

    return AggRangeTree


SumRangeTree = make_agg_meta(lambda a, b: a + b, lambda a, b: a - b)
