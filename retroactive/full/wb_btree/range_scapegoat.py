"""A dynamic range tree based on the scapegoat tree."""
from typing import (TypeVar, Generic, Optional, List, Tuple,
                    Generator, Callable, Iterable, Any)
from abc import ABC, abstractmethod
from copy import copy
from math import log
from collections import deque

K = TypeVar('K')
V = TypeVar('V')
MetaType = Optional[Any]  # TODO: improve
KVMeta = Tuple[K, V, MetaType]
NodeType = 'RangeNode[K, V, Meta]'
RebuildType = Callable[[NodeType], NodeType]
NodeIterator = Generator[NodeType, None, None]
KVMetaIterator = Generator[KVMeta, None, None]


class RangeMeta(ABC):
    """Abstract base class for range metadata."""
    @abstractmethod
    def __init__(self, key: K, val: V, container: NodeType):
        """Initializes range metadata."""

    @abstractmethod
    def insert(self, key: K, val: V, container: NodeType) -> None:
        """Updates range metadata to reflect an insert of a leaf."""

    @abstractmethod
    def remove(self, key: K, val: V, container: NodeType) -> None:
        """Updates range metadata to reflect removal of a leaf."""


class RangeTree(Generic[K, V]):
    """A dynamic range tree based on the scapegoat tree.

    TODO: more documentation here.
    """
    def __init__(self,
                 rebuild_fn: RebuildType,
                 meta_cls: MetaType = None):
        """Creates a weight-balanced B-tree with balance factor `d`."""
        self._meta_cls = meta_cls
        self._rebuild_fn = rebuild_fn
        self.root: Optional[NodeType] = None

    def find(self, key: K) -> Optional[V]:
        if self.root:
            return self.root.find(key)

    def insert(self, key: K, val: V) -> None:
        """Inserts a key-value pair into the tree.

        If `key` is already in the tree, a `ValueError`` is raised."""
        if self.root:
            self.root = self.root.insert(key, val, self._rebuild_fn)
        else:
            self.root = RangeNode(key, val, self._meta_cls)

    def remove(self, key: K) -> None:
        """Remove a key-value pair from the tree.

        If the `key` is not in the tree, a ``ValueError`` is raised."""
        if self.root:
            self.root = self.root.remove(key, self._rebuild_fn)
        else:
            raise ValueError('Cannot remove from an empty tree.')

    def all(self) -> KVMetaIterator:
        """Returns all key-value pairs in order."""
        if self.root:
            return self.root.all()

    def check_invariants(self) -> None:
        """Verifies that the tree is well-formed."""
        # TODO


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
        """Finds a path to a (possibly nonexistent) key."""
        yield self
        if not self.is_leaf:
            if self.left and self.left.max >= key:
                yield from self.left.path(key)
            elif self.right and self.right.min <= key:
                yield from self.right.path(key)

    def insert(self,
               key: K,
               val: V,
               rebuild_fn: RebuildType,
               alpha: float = 2/3) -> Tuple[NodeType, int]:
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
        if len(path) > log(self.ub) / log(1 / alpha):
            scapegoat = None
            parent = None
            for parent, child in zip(reversed(path[:-1]),
                                     reversed(path[1:])):
                if scapegoat:
                    break
                if child.size > alpha * parent.size:
                    scapegoat = parent
            if scapegoat == self:
                rebuilt = rebuild_fn(self)
                return rebuilt
            if scapegoat == parent.left:
                parent.left = rebuild_fn(scapegoat)
            else:
                parent.right = rebuild_fn(scapegoat)
        return self

    def remove(self, key: K, rebuild_fn: RebuildType) -> Optional[NodeType]:
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
        yield from self.left.all_nodes()
        yield from self.right.all_nodes()

    def all(self) -> KVMetaIterator:
        """Finds all the key-value pairs in the subtree rooted at the node."""
        print(self, self.left, self.right)
        if self.is_leaf:
            yield (self.min, self.val, self.meta)
        else:
            # In-order traversal.
            yield from self.left.all()
            yield from self.right.all()

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
    aggregation (e.g. +, *)."""
    class AggMeta(RangeMeta):
        def __init__(self, key: K, val: V, container: NodeType):
            super().__init__(key, val, container)
            self.val = val

        def insert(self, key: K, val: V, container: NodeType) -> None:
            self.val = insert_fn(self.val, val)

        def remove(self, key: K, val: V, container: NodeType) -> None:
            self.val = remove_fn(self.val, val)

        def __repr__(self):
            return str(self.val)

    def rebuild_fn(root: NodeType) -> NodeType:
        """Rebuild from the bottom up."""
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
                # The key and container parameters aren't actually used here.
                parent.meta = AggMeta(left.min, left.meta.val, left)
                parent.meta.insert(right.max, right.meta.val, right)
                next_level.append(parent)
            else:
                next_level.append(left)
            if not leaves and len(next_level) > 1:
                leaves = next_level
                next_level = deque()
        return next_level.pop()


    class AggRangeTree(RangeTree):
        def __init__(self):
            super().__init__(rebuild_fn, AggMeta)

    return AggRangeTree


SumRangeTree = make_agg_meta(lambda a, b: a + b, lambda a, b: a - b)
