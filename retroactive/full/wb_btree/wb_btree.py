"""Strongly weight-balanced B-trees.

Based on:
    [1] Lars Arge and Jeffrey Scott Vitter, Optimal external memory
        interval management, SIAM Journal on Computing, 32 (2003),
        pp. 1488–1508. (http://www.ittc.ku.edu/~jsv/Papers/
        ArV03.interval_managementOfficial.pdf)
    [2]  Bender, M.A., Demaine, E.D., Farach-Colton, M.: Cache-oblivious
         b-trees. In: Proceedings of the 41st Annual Symposium on
         Foundations of Computer Science, pp. 399–409. IEEE (2000).
         (http://supertech.csail.mit.edu/papers/DemaineKaLiSi15.pdf)
    (TODO: pick a citation style)
"""
from typing import TypeVar, Generic, Optional, List, Tuple, Generator
from collections import deque

K = TypeVar('K')
V = TypeVar('V')
KV = Tuple[K, V]
NodeIterator = Generator['WBBNode[K, V]', None, None]
KVIterator = Generator[KV, None, None]


class WBBTree(Generic[K, V]):
    """A weight-balanced B-tree."""
    def __init__(self, d: int = 8):
        """Creates a weight-balanced B-tree with balance factor `d`."""
        # Arge and Vitter distinguish between the branching parameter `a`
        # and a leaf parameter `k`. Following Bender et al., we use k = 1.
        if d <= 4:
            raise ValueError('Balance factor must be >4.')
        self.d = d
        self.deleted = 0
        self.root: WBBNode[K, V] = WBBNode(d=d)

    def find(self, key: K) -> Optional[V]:
        """Searches for a key in the tree.

        If the key is found, its associate value is returned. Otherwise,
        `None` is returned."""
        return self.root.find(key)

    def insert(self, key: K, val: V) -> None:
        """Inserts a key-value pair into the tree.

        If `key` is already in the tree, a `ValueError`` is raised."""
        self.root = self.root.insert(key, val)

    def remove(self, key: K) -> None:
        """Removes a key-value pair from the tree.

        If the `key` is not in the tree, a ``ValueError`` is raised."""
        self.root.mark_deleted(key)
        self.deleted += 1
        if self.deleted > self.root.weight // 2:
            # Globally rebalance.
            # TODO: use a different order here?
            new_root: WBBNode[K, V] = WBBNode(d=self.d)
            for k, v in self.root.all():
                new_root.insert(k, v)
            self.root = new_root
            self.deleted = 0

    def all(self) -> KVIterator:
        """Returns all key-value pairs in order."""
        return self.root.all()

    def min(self) -> Optional[KV]:
        """Finds the minimum key-value pair in the tree."""
        return self.root.min()

    def max(self) -> Optional[KV]:
        """Finds the maximum key-value pair in the tree."""
        return self.root.max()

    @property
    def size(self) -> int:
        """The number of key-value pairs in the tree."""
        return self.root.weight - self.deleted

    def _depth_invariant(self) -> bool:
        """Invariant 1: all leaves of the tree have the same depth."""
        return len(set(depth for depth, _ in self.root.leaves())) == 1

    def _root_invariant(self) -> bool:
        """Invariant 2: the root has between 2 and 4d children."""
        return self.root.is_leaf or 2 <= len(self.root.children) <= 4 * self.d

    def _weight_invariant(self) -> bool:
        """Invariant: a node's weight == the number of keys it contains +
        the weight of its children."""
        return all(node.weight == len(node.keys) + sum(c.weight
                                                       for c in node.children)
                   for node in self.root.all_nodes())

    def _size_invariant(self) -> bool:
        """Invariant: a node's size == the number of non-deleted keys it
        contains + the sizes of its children."""
        return all(node.size == len(node.keys) - len(node.deleted) +
                   sum(c.size for c in node.children)
                   for node in self.root.all_nodes())

    def _kv_invariant(self) -> bool:
        """Invariant: number of node keys == number of node values."""
        return all(
            len(node.keys) == len(node.vals) for node in self.root.all_nodes())

    def _balance_invariant(self) -> bool:
        """Invariant 3: balance.

        From Bender et al.:
            Consider a nonroot node u at height h in the tree. (Leaves have
            height 1.) The weight w(u) of u is the number of nodes in the
            subtree rooted at u. This weight is bounded by
            d^{h-1} / 2 ≤ w(u) ≤ 2d^{h-1}.
        """
        leaf_depth, _ = next(self.root.leaves())
        height = leaf_depth + 2  # We assume the depth invariant holds.
        nodes = deque((height - 1, c) for c in self.root.children)
        while nodes:
            node_height, node = nodes.popleft()
            if node.weight < self.d**(node_height - 1) / 2 or \
               node.weight > 2 * self.d**(node_height - 1):
                return False
            for child in node.children:
                nodes.append((node_height - 1, child))
        return True

    def _local_balance_invariant(self) -> bool:
        """Invariant 5: All internal non-root nodes have between d/4 and 4d
        children."""
        return all(self.d // 4 <= len(node.children) <= 4 *
                   self.d or node == self.root
                   for node in self.root.internal_nodes())

    def _num_children_invariant(self) -> bool:
        """Invariant: At each internal node, # of children == # of keys + 1."""
        return all(
            len(node.children) == len(node.keys) + 1
            for node in self.root.internal_nodes())

    def _num_keys_invariant(self) -> bool:
        """Invariant: Each node has ≥1 key."""
        return all(len(node.keys) > 0 for node in self.root.all_nodes()) or \
               self.root.weight == 0

    def _key_order_invariant(self) -> bool:
        """Invariant: At each node, keys are in order."""
        return all(node.keys == sorted(node.keys)
                   for node in self.root.all_nodes())

    def _deleted_invariant(self) -> bool:
        """Invariant: at most half of the keys in the tree are deleted."""
        return self.deleted <= self.root.weight // 2

    def check_invariants(self) -> None:
        """Verifies that the tree is well-formed."""
        assert self._depth_invariant(), 'Leaves have unequal depths.'
        assert self._root_invariant(), 'Root has the wrong number of children.'
        assert self._weight_invariant(), '≥1 node has the wrong weight.'
        assert self._size_invariant(), '≥1 node has the wrong size.'
        assert self._num_children_invariant(), \
               'At ≥1 node, # of children ≠ # of keys + 1.'
        assert self._kv_invariant(), \
                'At ≥1 node, # of keys ≠ # of values.'
        assert self._num_keys_invariant(), \
                '≥1 node has no keys.'
        assert self._local_balance_invariant(), \
               '≥1 internal node has < d/4 or > 4d children.'
        assert self._balance_invariant(), 'Tree is unbalanced.'
        assert self._key_order_invariant(), \
               '≥1 node has keys in the wrong order.'
        assert self._deleted_invariant(), 'Tree has too many deleted nodes.'


class WBBNode(Generic[K, V]):
    """A node in a weight-balanced B-tree."""
    def __init__(self, d: int = 8):
        """Creates a weight-balanced B-tree node with balance factor `d`."""
        self.weight = 0  # number of slots used
        self.size = 0    # number of currently inserted elements
        self.d = d
        self.deleted: List[K] = []
        self.keys: List[K] = []
        self.vals: List[V] = []
        self.children: List[WBBNode[K, V]] = []

    @property
    def is_leaf(self) -> bool:
        """Does the node have any children?"""
        return len(self.children) == 0

    def find(self, key: K) -> Optional[V]:
        """Looks up a key in the subtree rooted at the node.

        Returns:
            The value associated with the key, if a match is found.
            Otherwise, returns `None`.
        """
        node = None
        for node in self.path(key):
            pass  # Find the last node on the path.
        try:
            if key in node.deleted:
                return None
            return node.vals[node.keys.index(key)]
        except ValueError:
            return None

    def path(self, key: K) -> NodeIterator:
        """Finds a path to a (possibly nonexistent) key."""
        yield self
        if not self.is_leaf and key not in self.keys:
            for child, child_key in zip(self.children, self.keys):
                if child_key > key:
                    yield from child.path(key)
                    return
            if len(self.children) == len(self.keys) + 1:
                yield from self.children[-1].path(key)

    def split(self) -> Tuple['WBBNode[K, V]', KV, 'WBBNode[K, V]']:
        """Finds a weight-balanced split of the subtree rooted at the node."""
        left: WBBNode[K, V] = WBBNode(d=self.d)
        idx = 0
        if self.is_leaf:
            # Base case: leaf (no children), so we split like a normal B-tree.
            idx = self.weight // 2
            left.weight = idx
        else:
            # Internal node: find an approximately weight-balanced split.
            data = zip(self.children, self.keys, self.vals)
            weight = 0
            for idx, (node, k, v) in enumerate(data):
                if weight + node.weight + 1 < self.weight / 2:
                    weight += node.weight + 1
                else:
                    break
        median_key = self.keys[idx]
        median_val = self.vals[idx]
        left.keys = self.keys[:idx]
        left.vals = self.vals[:idx]
        left.children = self.children[:idx + 1]
        left.deleted = [k for k in self.deleted if k < median_key]
        left.weight = sum(c.weight for c in left.children) + len(left.keys)
        left.size = sum(c.size for c in left.children) + len(left.keys)
        left.size -= len(left.deleted)
        right: WBBNode[K, V] = WBBNode(d=self.d)
        right.children = self.children[idx + 1:]
        right.keys = self.keys[idx + 1:]
        right.vals = self.vals[idx + 1:]
        right.deleted = [k for k in self.deleted if k > median_key]
        right.weight = sum(c.weight for c in right.children) + len(right.keys)
        right.size = sum(c.size for c in right.children) + len(right.keys)
        right.size -= len(right.deleted)
        return left, (median_key, median_val), right

    def insert(self, key: K, val: V) -> 'WBBNode[K, V]':
        """Inserts a key-value pair into the subtree rooted at the node.

        If the key already exists, a `ValueError` is raised.

        Returns:
            A new root with the key inserted.
        """
        path = list(self.path(key))
        leaf = path[-1]
        if key in leaf.keys:
            if key in leaf.deleted:
                # Just unmark the key and replace its value!
                # TODO: update global deletion count?
                leaf.deleted.remove(key)
                leaf.vals[leaf.keys.index(key)] = val
                return self
            raise ValueError(f'Key "{key}" already in tree')

        # Insert the new key-value pair in place.
        inserted = False
        for idx, sibling_key in enumerate(leaf.keys):
            if sibling_key > key:
                leaf.keys.insert(idx, key)
                leaf.vals.insert(idx, val)
                inserted = True
                break
        if not inserted:
            leaf.keys.append(key)
            leaf.vals.append(val)
        for node in path:
            node.weight += 1
            node.size += 1

        # Move back up the tree, splitting as necessary.
        for level, child in enumerate(reversed(path)):
            target_weight = self.d**(level + 1)
            if child.weight > 2 * target_weight:
                left, median, right = child.split()
                if level + 1 == len(path):
                    # Splitting at the top requires a new root node.
                    new_root: WBBNode[K, V] = WBBNode(d=self.d)
                    new_root.insert(median[0], median[1])
                    new_root.children = [left, right]
                    new_root.weight = left.weight + right.weight + 1
                    new_root.size = left.size + right.size + 1
                    return new_root
                # Otherwise, replace the child node with the new left node and
                # insert the right node next to it.
                parent = path[len(path) - level - 2]
                for idx, node in enumerate(parent.children):
                    if node == child:
                        parent.children[idx] = left
                        parent.keys.insert(idx, median[0])
                        parent.vals.insert(idx, median[1])
                        parent.children.insert(idx + 1, right)
                        break
        return self

    def mark_deleted(self, key: K) -> 'WBBNode[K, V]':
        """Marks a key as deleted (lazy deletion).

        If the key does not exist, a `ValueError` is raised."""
        path = list(self.path(key))
        if key not in path[-1].keys or key in path[-1].deleted:
            raise ValueError(f'Cannot delete "{key}" (not in tree)')
        path[-1].deleted.append(key)
        for node in path:
            node.size -= 1

    def leaves(self) -> Generator[Tuple[int, 'WBBNode[K, V]'], None, None]:
        """Finds all the leaves in the subtree rooted at the node."""
        if self.is_leaf:
            yield (0, self)
        else:
            for child in self.children:
                yield from ((level + 1, node)
                            for level, node in child.leaves())

    def internal_nodes(self) -> NodeIterator:
        """Finds all the internal nodes in the subtree rooted at the node."""
        if not self.is_leaf:
            yield self
            for child in self.children:
                yield from child.internal_nodes()

    def all_nodes(self) -> NodeIterator:
        """Finds all the nodes in the subtree rooted at the node."""
        yield self
        for child in self.children:
            yield from child.all_nodes()

    def all(self) -> KVIterator:
        """Finds all the key-value pairs in the subtree rooted at the node."""
        if self.is_leaf:
            for key, val in zip(self.keys, self.vals):
                if key not in self.deleted:
                    yield (key, val)
        else:
            # In-order traversal.
            for child, key, val in zip(self.children, self.keys, self.vals):
                yield from child.all()
                if key not in self.deleted:
                    yield (key, val)
            yield from self.children[-1].all()

    def min(self) -> Tuple[K, V]:
        """Returns the minimum key-value pair in the subtree."""
        if self.is_leaf:
            for (k, v) in zip(self.keys, self.vals):
                if k not in self.deleted:
                    return (k, v)
        for child in self.children:
            if child.size > 0:
                return child.min()

    def max(self) -> Tuple[K, V]:
        """Returns the maximum key-value pair in the subtree."""
        if self.is_leaf:
            for (k, v) in zip(reversed(self.keys), reversed(self.vals)):
                if k not in self.deleted:
                    return (k, v)
        for child in reversed(self.children):
            if child.size > 0:
                return child.max()

    def __repr__(self):
        if self.is_leaf:
            if self.weight == 0:
                return 'empty leaf node'
            if self.weight == 1:
                return f'singleton leaf node with key {self.keys[0]}'
            return (f'leaf node (weight {self.weight}) with keys: ' +
                    ' '.join(str(k) for k in self.keys))
        return (f'internal node ({len(self.children)} children, weight ' +
                f'{self.weight}) with keys: ' +
                ' '.join(str(k) for k in self.keys))
