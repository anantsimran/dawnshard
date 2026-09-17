"""Binary max-heap of (priority, item) pairs with O(log n) priority updates.

Python 3.12+ (uses PEP 695 generic syntax).
"""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrioritizedItem[T: Hashable]:
    """One heap entry.

    Frozen so callers cannot change a priority behind the heap's back,
    which would silently break the heap invariant.

    Attributes:
        priority: Higher values pop first.
        item: Unique, hashable identifier of the entry.
    """

    priority: int
    item: T


class MaxHeap[T: Hashable]:
    """Array-backed binary max-heap with an item -> index map.

    Invariant: heap[i].priority >= priority of children 2i+1 and 2i+2.
    The index map lets update_priority find an item in O(1) instead of O(n).
    Ties pop in arbitrary order.
    """

    def __init__(self, items: Iterable[PrioritizedItem[T]]) -> None:
        """Heapify items in O(n) (Floyd's bottom-up construction).

        Args:
            items: Entries to load. Copied, so the caller's list is not reordered.

        Raises:
            ValueError: If two entries share an item.
        """
        self._heap = list(items)  # noqa: NAR001
        self._index = {p.item: i for i, p in enumerate(self._heap)}  # noqa: NAR001
        if len(self._index) != len(self._heap):  # noqa: NAR001
            raise ValueError("items must be unique")  # noqa: NAR001
        for i in reversed(range(len(self._heap) // 2)):  # noqa: NAR001
            self._sift_down(i=i)

    def __len__(self) -> int:
        """Return the number of items, enabling `while heap:` loops."""
        return len(self._heap)  # noqa: NAR001

    def __contains__(self, item: object) -> bool:
        """Return True if item is in the heap, enabling `item in heap`, in O(1)."""
        return item in self._index

    def size(self) -> int:
        """Return the number of items; same as len(heap)."""
        return len(self._heap)  # noqa: NAR001

    def add_item(self, item: T, priority: int) -> None:
        """Insert a new item in O(log n).

        Args:
            item: New item, not already in the heap.
            priority: Initial priority.

        Raises:
            ValueError: If item is already in the heap.
        """
        if item in self._index:
            raise ValueError("item already in heap")  # noqa: NAR001
        self._heap.append(PrioritizedItem(priority=priority, item=item))  # noqa: NAR001
        self._index[item] = len(self._heap) - 1  # noqa: NAR001
        self._sift_up(i=self._index[item])

    def pop(self) -> PrioritizedItem[T]:
        """Remove and return the highest-priority entry in O(log n).

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty heap")  # noqa: NAR001
        return self.remove(item=self._heap[0].item)

    def remove(self, item: T) -> PrioritizedItem[T]:
        """Remove and return the entry for an existing item in O(log n).

        Args:
            item: Item currently in the heap.

        Raises:
            KeyError: If item is not in the heap.
        """
        i = self._index.pop(item)  # noqa: NAR001
        removed, last = self._heap[i], self._heap.pop()
        if i < len(self._heap):  # noqa: NAR001
            # The last entry fills the hole and may belong above or below it.
            self._heap[i] = last
            self._index[last.item] = i
            if last.priority > removed.priority:
                self._sift_up(i=i)
            else:
                self._sift_down(i=i)
        return removed

    def get_priority(self, item: T) -> int:
        """Return the current priority of an existing item in O(1).

        Args:
            item: Item currently in the heap.

        Raises:
            KeyError: If item is not in the heap.
        """
        return self._heap[self._index[item]].priority

    def update_priority(
        self,
        item: T,
        absolute: int | None = None,
        relative: int | None = None,
    ) -> None:
        """Raise or lower the priority of an existing item in O(log n).

        Exactly one of absolute or relative must be given.

        Args:
            item: Item currently in the heap.
            absolute: New priority.
            relative: Amount added to the current priority; negative lowers it.

        Raises:
            KeyError: If item is not in the heap.
            ValueError: If both or neither of absolute and relative are given.
        """
        i = self._index[item]
        current = self._heap[i].priority
        if absolute is not None and relative is None:
            priority = absolute
        elif relative is not None and absolute is None:
            priority = current + relative
        else:
            raise ValueError("pass exactly one of absolute or relative")  # noqa: NAR001
        self._heap[i] = PrioritizedItem(priority=priority, item=item)
        if priority > current:
            self._sift_up(i=i)
        else:
            self._sift_down(i=i)

    def _sift_up(self, i: int) -> None:
        """Swap entry i with its parent until the invariant holds.

        Keeps the index map in sync.

        Args:
            i: Index of the entry that may be larger than its parent.
        """
        heap = self._heap
        while i > 0 and heap[i].priority > heap[parent := (i - 1) // 2].priority:
            heap[i], heap[parent] = heap[parent], heap[i]
            self._index[heap[i].item] = i
            self._index[heap[parent].item] = parent
            i = parent

    def _sift_down(self, i: int) -> None:
        """Swap entry i with its larger child until the invariant holds.

        Keeps the index map in sync.

        Args:
            i: Index of the entry that may be smaller than its children.
        """
        heap, size = self._heap, len(self._heap)  # noqa: NAR001
        while (child := 2 * i + 1) < size:
            if child + 1 < size and heap[child + 1].priority > heap[child].priority:
                child += 1
            if heap[child].priority <= heap[i].priority:
                return
            heap[i], heap[child] = heap[child], heap[i]
            self._index[heap[i].item] = i
            self._index[heap[child].item] = child
            i = child
