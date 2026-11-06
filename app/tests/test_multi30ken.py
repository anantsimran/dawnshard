import pytest
import torch
from training.constants import CROSS_ENTROPY_IGNORE_INDEX
from training.dataload import multi30k
from training.dataload.multi30k import _mlm_collate, _translation_collate
from training.preprocess.bpe.tokenizer import BpeTokenizer
from training.preprocess.constants import MASK_ID, PAD_ID, PROTECTED_IDS

# Bytes 0-255, the protected ids, then two merge ids.
VOCAB_LENGTH = 256 + len(PROTECTED_IDS) + 2  # noqa: NAR001


def _rows(num_rows, seq_len):
    ids = torch.randint(low=0, high=256, size=(num_rows, seq_len), generator=torch.Generator())
    ids[:, seq_len // 2 :] = PAD_ID
    return list(enumerate(ids.unbind(dim=0)))  # noqa: NAR001


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


def test_translation_collate_stacks_x_and_flattens_y_row_major():
    inputs = torch.randint(low=0, high=256, size=(3, 2, 8))
    targets = torch.randint(low=0, high=256, size=(3, 8))

    x, y = _translation_collate(rows=list(zip(inputs, targets)))  # noqa: NAR001

    assert torch.equal(x, inputs)  # noqa: NAR001
    assert torch.equal(y, targets.reshape(-1))  # noqa: NAR001


def _fake_texts(monkeypatch, texts_by_lang):
    monkeypatch.setattr(
        target=multi30k,
        name="get_multi30k_texts",
        value=lambda lang, cache=None: texts_by_lang[lang],
    )


def _texts(count, prefix):
    rows = [f"{prefix} sentence {i}" for i in range(count)]
    return multi30k.Multi30kTexts(train=rows, val=rows[:4], test=rows[:3])


def test_translation_loader_keeps_pairs_aligned_after_sampling(monkeypatch):
    _fake_texts(monkeypatch, {"en": _texts(20, "en"), "de": _texts(20, "de")})
    torch.manual_seed(0)
    train_loader, val_loader, test_loader = multi30k.get_multi30k_translation_dataloader(
        seq_len=24,
        batch_size=20,
        src_tokenizer=BpeTokenizer(merges=[]),
        tgt_tokenizer=BpeTokenizer(merges=[]),
        train_num=8,
        val_num=3,
        test_num=2,
    )
    x, _ = next(iter(train_loader))
    for src, tgt_in in x.tolist():
        src_text = bytes(i for i in src if 0 < i < 256).decode()
        tgt_text = bytes(i for i in tgt_in if 0 < i < 256).decode()
        assert src_text.split()[-1] == tgt_text.split()[-1]
    assert (len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)) == (
        8,
        3,
        2,
    )


def test_translation_loader_rejects_languages_with_different_row_counts(monkeypatch):
    _fake_texts(monkeypatch, {"en": _texts(20, "en"), "de": _texts(19, "de")})
    with pytest.raises(ValueError, match="cannot be aligned"):
        multi30k.get_multi30k_translation_dataloader(
            seq_len=8,
            batch_size=4,
            src_tokenizer=BpeTokenizer(merges=[]),
            tgt_tokenizer=BpeTokenizer(merges=[]),
        )


def test_loaders_draw_once_from_the_torch_rng_even_without_train_num(monkeypatch):
    _fake_texts(monkeypatch, {"en": _texts(10, "en")})
    for train_num in (None, 5):
        torch.manual_seed(0)
        torch.rand(size=())
        expected = torch.rand(size=(1,))[0]
        torch.manual_seed(0)
        multi30k.get_multi30ken_mlm_dataloader(
            seq_len=8,
            val_seed=0,
            test_seed=100,
            batch_size=4,
            tokenizer=BpeTokenizer(merges=[]),
            train_num=train_num,
        )
        assert torch.rand(size=(1,))[0] == expected
