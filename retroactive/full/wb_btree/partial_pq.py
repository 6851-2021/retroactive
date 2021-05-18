"""A partially retroactive priority queue.

This queue can be used independently, but it is intended to be
nested within a fully retroactive priority queue.
"""
from typing import TypeVar, Generic, Union, Tuple, Optional
from range_scapegoat import SumRangeTree
from wb_btree import WBBTree
from insert_tree import InsertTree
from prefix_sum_tree import PrefixSumTree

TS = float
TS_ZERO = 0.0
TS_EPSILON = 1
V = TypeVar('V')

INSERT = 'insert'
DELETE_MIN = 'delete-min'

Event = str
InsertEvent = Tuple[Event, V]
DeleteMinEvent = Event
EventType = Union[InsertEvent, DeleteMinEvent]


class PRPriorityQueue(Generic[V]):
    def __init__(self):
        self.t_max = TS_ZERO
        self.t_next = TS_ZERO
        self.now: WBBTree[V, TS] = WBBTree()  # Qnow
        self.deleted: WBBTree[V, TS] = WBBTree()  # Qdel
        self.events: WBBTree[TS, EventType] = WBBTree()
        self.inserts: InsertTree[TS, V] = InsertTree()
        self.updates: PrefixSumTree[TS, int] = PrefixSumTree(0)

    def insert(self, val: V, t: Optional[TS] = None) -> None:
        """Inserts a value in the priority queue at time `t`.

        If `t` is not specified, the value is inserted strictly
        after the latest event time in the queue."""
        if t is None:
            self.t_next += TS_EPSILON
            t = self.t_next
        elif t <= TS_ZERO:
            raise ValueError(f'timestamp must be > {TS_ZERO}.')
        self.events.insert(t, (INSERT, val))

        bridge = self.updates.last_node_with_sum(t, 0)
        if bridge is None:
            t_bridge = TS_ZERO
        else:
            t_bridge = bridge.min
        absent_val, absent_t = self.inserts.max_absent_in_range(
            t_bridge, self.t_max)
        if val == absent_val:
            raise ValueError(f'Value {val} not unique.')
        if absent_val is None or val > absent_val:
            self.now.insert(val, t)
            self.inserts.insert(t, val)
            self.updates.insert(t, 0)
            self.inserts.mark_present(t)
        else:
            self.now.insert(absent_val, absent_t)
            self.inserts.insert(t, val)
            self.inserts.mark_absent(t)
            self.inserts.mark_present(absent_t)
            self.updates.insert(t, 1)
            self.deleted.insert(val, t)
        self.t_max = max(t, self.t_max)

    def delete_min(self, t: Optional[TS] = None) -> None:
        """Inserts a delete-min operation at time `t`."""
        if t is None:
            self.t_next += TS_EPSILON
            t = self.t_next
        elif t <= TS_ZERO:
            raise ValueError(f'timestamp must be > {TS_ZERO}.')

        bridge = self.updates.first_node_with_sum(t, 0)
        if bridge is None:
            t_bridge = self.t_max
        else:
            t_bridge = bridge.min
        present_val, present_t = self.inserts.min_present_in_range(
            TS_ZERO, t_bridge)
        self.events.insert(t, DELETE_MIN)
        self.updates.insert(t, -1)
        if present_t is not None:
            self.updates.remove(present_t)
            self.updates.insert(present_t, 1)
            self.now.remove(present_val)
            self.inserts.mark_absent(present_t)
            self.deleted.insert(present_val, present_t)

    def delete_op(self, t: TS) -> None:
        """Deletes the operation at time `t` from the queue.

        If no event exists at time `t`, a `ValueError` is raised.
        """
        event = self.events.find(t)
        if event is None:
            raise ValueError(f'No event at time {t}.')
        if event == DELETE_MIN:
            self._delete_delete_min(t)
        else:
            self._delete_insert(t)
        max_event = self.events.max()
        if max_event is None:
            self.t_max = TS_ZERO
        else:
            self.t_max = max_event[0]

    def _delete_delete_min(self, t: TS) -> None:
        """Deletes a delete-min operation at time `t`."""
        bridge = self.updates.last_node_with_sum(t, 0)
        if bridge is None:
            t_bridge = TS_ZERO
        else:
            t_bridge = bridge.min
        absent_val, absent_t = self.inserts.max_absent_in_range(
            t_bridge, self.t_max)
        self.events.remove(t)
        self.now.insert(absent_val, absent_t)
        self.inserts.mark_present(absent_t)
        self.updates.remove(absent_t)
        self.updates.insert(absent_t, 0)
        self.deleted.remove(absent_val)

    def _delete_insert(self, t: TS) -> None:
        """Deletes an insert operation at time `t`."""
        val = self.inserts.find(t)
        self.events.remove(t)
        if self.now.find(val):
            # Case: The element to delete is still in Qnow.
            self.now.remove(val)
            self.inserts.remove(t)
            self.updates.remove(t)
            self.deleted.insert(val, t)
        else:
            # Case: The element to delete is now longer in Qnow.
            bridge = self.updates.first_node_with_sum(t, 0)
            if bridge is None:
                t_bridge = self.t_max
            else:
                t_bridge = bridge.min
            present_val, present_t = self.inserts.min_present_in_range(
                TS_ZERO, t_bridge)
            self.now.remove(present_val)
            self.inserts.remove(present_t)
            self.updates.remove(present_t)
            self.deleted.insert(present_val, present_t)

    def __repr__(self):
        status = 'Qnow: ' + ' '.join([str(k) for k, _ in self.now.all()])
        status += '\nevents:\n'
        for t, event in self.events.all():
            if event == DELETE_MIN:
                status += f'{t}: delete min\n'
            else:
                status += f'{t}: insert {event[1]}\n'
        return status
