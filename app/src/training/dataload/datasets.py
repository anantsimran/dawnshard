"""Dataset classes shared by more than one loader."""

from collections.abc import Sequence

from torch.utils.data import Dataset


class InMemoryDataset[T](Dataset[T]):
    """One split held in memory as a sequence of already-built items of type T.

    The dataset returns each item as stored and does not split it into input and
    target; the collate_fn builds the batch from a list of items. A custom collate_fn
    decides the (X, Y) structure itself, so it can derive the target from a single
    tensor, as one that masks tokens does. A collate_fn that needs to know which sample
    it holds, for example to seed it, needs the index stored in the item: build the
    items as enumerate(...), so T = (index, ...).

    PyTorch's default collate_fn decides nothing. It keeps the item's structure and
    stacks each leaf along a new B axis, so (X, Y) must already be in T. Store
    T = (x, y), and one sample and the batch built from B of them look like this:

    ```
    sample: (x, y)  x (L,), y ()
    batch:  [X, Y]  X (B, L), Y (B,)
    ```

    For this to work, T must be one of these, with the same structure in every item:

    - A tensor or NumPy array of the same shape in every item, stacked to (B, ...).
      Variable-length items must be padded first, or stack raises a RuntimeError.
    - An int or float, which becomes a (B,) int64 or float64 tensor. A str or bytes
      stays a list.
    - A tuple, list, dict, or namedtuple of the above, with the same length or keys in
      every item. Each position is collated on its own; a tuple comes back as a list,
      a dict or namedtuple keeps its type.
    """

    def __init__(self, items: Sequence[T]) -> None:
        """Wrap items as they are; nothing is copied or transformed."""
        super().__init__()
        self._items = items

    def __getitem__(self, index: int) -> T:
        """Return the item at index, as stored."""
        return self._items[index]

    def __len__(self) -> int:
        """Return the number of items in this split."""
        return len(self._items)  # noqa: NAR001
