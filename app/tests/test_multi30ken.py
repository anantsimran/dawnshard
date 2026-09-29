import pytest
import torch
from training.common.constants import MASK_ID, PAD_ID, PROTECTED_IDS
from training.constants import CROSS_ENTROPY_IGNORE_INDEX
from training.dataload.multi30ken import Multi30kENDataset, _mlm_collate

# Bytes 0-255, the protected ids, then two merge ids.
VOCAB_LENGTH = 256 + len(PROTECTED_IDS) + 2  # noqa: NAR001


def _rows(num_rows, seq_len):
    ids = torch.randint(
        low=0, high=256, size=(num_rows, seq_len), generator=torch.Generator()
    )
    ids[:, seq_len // 2 :] = PAD_ID
    dataset = Multi30kENDataset(ids=ids)
    return [dataset[i] for i in range(len(dataset))]  # noqa: NAR001


def test_mlm_collate_returns_batch_shaped_x_and_flat_y():
    x, y = _mlm_collate(rows=_rows(num_rows=3, seq_len=8), vocab_length=VOCAB_LENGTH)

    assert x.shape == (3, 8)
    assert y.shape == (24,)
    assert x.dtype == y.dtype == torch.int64


def test_mlm_collate_flattens_y_row_major():
    rows = _rows(num_rows=4, seq_len=8)

    _, y = _mlm_collate(rows=rows, vocab_length=VOCAB_LENGTH, seed=3)

    # Row b alone, collated with the same per-sample seed, is y[b*8 : (b+1)*8].
    for b, row in enumerate(rows):  # noqa: NAR001
        _, alone = _mlm_collate(rows=[row], vocab_length=VOCAB_LENGTH, seed=3)
        assert torch.equal(y[b * 8 : (b + 1) * 8], alone)  # noqa: NAR001


def test_mlm_collate_with_seed_repeats_masks_on_every_call():
    rows = _rows(num_rows=64, seq_len=16)

    first = _mlm_collate(rows=rows, vocab_length=VOCAB_LENGTH, seed=7)
    torch.rand(size=(5,))  # advance the global RNG between calls
    second = _mlm_collate(rows=rows, vocab_length=VOCAB_LENGTH, seed=7)

    assert torch.equal(first[0], second[0])  # noqa: NAR001
    assert torch.equal(first[1], second[1])  # noqa: NAR001


def test_mlm_collate_with_seed_does_not_consume_global_rng():
    rows = _rows(num_rows=64, seq_len=16)
    state = torch.get_rng_state()

    _mlm_collate(rows=rows, vocab_length=VOCAB_LENGTH, seed=7)

    assert torch.equal(torch.get_rng_state(), state)  # noqa: NAR001


def test_mlm_collate_changes_only_selected_positions_and_targets_their_ids():
    rows = _rows(num_rows=256, seq_len=16)
    original = torch.stack(tensors=[ids for _, ids in rows])

    x, y = _mlm_collate(rows=rows, vocab_length=VOCAB_LENGTH, seed=0)

    y = y.reshape(x.shape)  # noqa: NAR001
    selected = y != CROSS_ENTROPY_IGNORE_INDEX
    assert selected.any()
    assert torch.equal(y[selected], original[selected])  # noqa: NAR001
    assert torch.equal(x[~selected], original[~selected])  # noqa: NAR001


def test_mlm_collate_never_selects_protected_ids():
    ids = torch.tensor(data=PROTECTED_IDS * 50).reshape(10, -1)  # noqa: NAR001
    rows = [(i, row) for i, row in enumerate(ids)]  # noqa: NAR001

    x, y = _mlm_collate(rows=rows, vocab_length=VOCAB_LENGTH, seed=0)

    assert torch.equal(x, ids)  # noqa: NAR001
    assert (y == CROSS_ENTROPY_IGNORE_INDEX).all()


def test_mlm_collate_follows_80_10_10_and_draws_no_protected_replacement():
    rows = _rows(num_rows=2000, seq_len=32)
    original = torch.stack(tensors=[ids for _, ids in rows])

    x, y = _mlm_collate(rows=rows, vocab_length=VOCAB_LENGTH, seed=0)

    selected = y.reshape(x.shape) != CROSS_ENTROPY_IGNORE_INDEX  # noqa: NAR001
    candidates = original != PAD_ID
    assert selected.sum() / candidates.sum() == pytest.approx(expected=0.15, abs=0.01)
    masked = selected & (x == MASK_ID)
    replaced = selected & (x != MASK_ID) & (x != original)
    assert masked.sum() / selected.sum() == pytest.approx(expected=0.8, abs=0.02)
    # A random draw equals the original id 1 time in VOCAB_LENGTH - 4.
    assert replaced.sum() / selected.sum() == pytest.approx(expected=0.1, abs=0.02)
    protected = torch.tensor(data=PROTECTED_IDS)
    assert not torch.isin(elements=x[replaced], test_elements=protected).any()
    assert (x[replaced] < VOCAB_LENGTH).all()


def test_mlm_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        _mlm_collate(rows=[], vocab_length=VOCAB_LENGTH)
