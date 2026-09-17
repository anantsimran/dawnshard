import pytest
from training.common.priority_queue import MaxHeap, PrioritizedItem


def _heap(pairs):
    return MaxHeap(items=(PrioritizedItem(priority=p, item=i) for i, p in pairs))


def test_pop_returns_items_in_decreasing_priority_order():
    heap = _heap([("a", 3), ("b", 1), ("c", 5), ("d", 2)])
    popped = [heap.pop().item for _ in range(4)]
    assert popped == ["c", "a", "d", "b"]


def test_len_and_contains():
    heap = _heap([("a", 1), ("b", 2)])
    assert len(heap) == 2
    assert heap.size() == 2
    assert "a" in heap
    assert "z" not in heap


def test_constructor_rejects_duplicate_items():
    with pytest.raises(ValueError):
        _heap([("a", 1), ("a", 2)])


def test_add_item_then_pop_highest_priority():
    heap = _heap([("a", 1)])
    heap.add_item(item="b", priority=5)
    assert len(heap) == 2
    assert heap.pop().item == "b"


def test_add_item_rejects_duplicate():
    heap = _heap([("a", 1)])
    with pytest.raises(ValueError):
        heap.add_item(item="a", priority=2)


def test_pop_from_empty_heap_raises():
    heap = _heap([])
    with pytest.raises(IndexError):
        heap.pop()


def test_remove_existing_item():
    heap = _heap([("a", 3), ("b", 1), ("c", 2)])
    removed = heap.remove(item="b")
    assert removed.item == "b"
    assert removed.priority == 1
    assert "b" not in heap
    assert len(heap) == 2


def test_remove_missing_item_raises_keyerror():
    heap = _heap([("a", 1)])
    with pytest.raises(KeyError):
        heap.remove(item="missing")


def test_get_priority():
    heap = _heap([("a", 7)])
    assert heap.get_priority(item="a") == 7


def test_update_priority_absolute_reorders_heap():
    heap = _heap([("a", 1), ("b", 2)])
    heap.update_priority(item="a", absolute=10)
    assert heap.get_priority(item="a") == 10
    assert heap.pop().item == "a"


def test_update_priority_relative_adjusts_current_value():
    heap = _heap([("a", 5)])
    heap.update_priority(item="a", relative=-3)
    assert heap.get_priority(item="a") == 2


def test_update_priority_requires_exactly_one_of_absolute_or_relative():
    heap = _heap([("a", 1)])
    with pytest.raises(ValueError):
        heap.update_priority(item="a")
    with pytest.raises(ValueError):
        heap.update_priority(item="a", absolute=1, relative=1)


def test_heap_invariant_holds_after_mixed_operations():
    heap = _heap([("a", 5), ("b", 3), ("c", 8), ("d", 1)])
    heap.add_item(item="e", priority=6)
    heap.update_priority(item="b", relative=10)
    heap.remove(item="d")
    popped = [heap.pop().item for _ in range(len(heap))]
    assert popped == ["b", "c", "e", "a"]
