import csv
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from training.dataload.ag_news import (
    AGNewsDataset,
    _encode_rows,
    _load_merges,
    _read_raw_rows,
    _read_split,
    _save_merges,
    _write_split,
)
from training.common import bpe
from training.dataload.constants import DATASETS_CACHE_DIR
from training.metrics.classification_metrics import (
    ClassificationMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.model.ag_news_classifier import MAX_LEN, AGNewsClassifier
from training.model.ag_news_classifier import H as NUM_HEADS
from training.model.ag_news_classifier import NUM_BLOCKS
from training.train.model import EpochSpec, TrainState
from training.train.train_loop import fit
from training.transformer.probes import save_attention_maps


def test_read_raw_rows_converts_label_to_zero_indexed_and_joins_title_and_description(
    tmp_path,
):
    path = tmp_path / "raw.csv"
    with open(file=path, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["1", "Title", "Description"]])  # noqa: NAR001

    assert _read_raw_rows(path=path) == [("Title Description", 0)]


def test_read_raw_rows_replaces_escaped_linebreaks_with_space(tmp_path):
    path = tmp_path / "raw.csv"
    with open(file=path, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["2", "Title", r"line1\line2"]])  # noqa: NAR001

    assert _read_raw_rows(path=path) == [("Title line1 line2", 1)]


def test_write_split_then_read_split_roundtrips(tmp_path):
    path = tmp_path / "split" / "rows.csv"
    rows = [("first", 0), ("second", 3)]

    _write_split(rows=rows, path=path)

    assert _read_split(path=path) == rows


def test_save_merges_then_load_merges_roundtrips(tmp_path):
    path = tmp_path / "bpe.json"
    merges, vocab = bpe.train(text="the cat sat on the mat", num_merges=20)

    _save_merges(merges=merges, num_merges=20, path=path)
    loaded = _load_merges(path=path, num_merges=20)

    assert loaded is not None
    loaded_merges, loaded_vocab = loaded
    assert [m.to_str(vocab=loaded_vocab) for m in loaded_merges] == [
        m.to_str(vocab=vocab) for m in merges
    ]
    assert loaded_vocab == vocab


def test_load_merges_returns_none_when_num_merges_changed(tmp_path):
    path = tmp_path / "bpe.json"
    merges, _ = bpe.train(text="the cat sat on the mat", num_merges=20)

    _save_merges(merges=merges, num_merges=20, path=path)

    assert _load_merges(path=path, num_merges=21) is None


def test_load_merges_returns_none_for_cache_without_first_merge_id(tmp_path):
    path = tmp_path / "bpe.json"
    path.write_text(json.dumps({"num_merges": 1, "merges": [[97, 97]]}))

    assert _load_merges(path=path, num_merges=1) is None


def test_load_merges_returns_none_when_file_missing(tmp_path):
    assert _load_merges(path=tmp_path / "missing.json", num_merges=1) is None


def test_encode_rows_pads_to_max_len_with_pad_id():
    dataset = _encode_rows(rows=[("ab", 0)], merges=[], max_len=5)

    assert isinstance(dataset, AGNewsDataset)
    assert len(dataset) == 1
    ids, label = dataset[0]
    pad = bpe.PAD_ID
    assert torch.equal(ids, torch.tensor([97, 98, pad, pad, pad]))
    assert label == 0


def test_encode_rows_truncates_to_max_len():
    dataset = _encode_rows(rows=[("abcd", 0)], merges=[], max_len=2)

    ids, _ = dataset[0]
    assert torch.equal(ids, torch.tensor([97, 98]))


def _padded_ids(batch_size, length):
    ids = torch.randint(low=0, high=bpe.PAD_ID, size=(batch_size, length))
    ids[:, -2:] = bpe.PAD_ID
    return ids


def test_classifier_forward_stores_detached_attention_map_per_block():
    torch.manual_seed(seed=0)
    model = AGNewsClassifier(vocab_size=bpe.PAD_ID + 1)
    model(_padded_ids(batch_size=3, length=12))  # noqa: NAR001
    assert len(model.attention_map) == NUM_BLOCKS  # noqa: NAR001
    for weights in model.attention_map:
        assert weights.shape == (3, NUM_HEADS, 12, 12)
        assert not weights.requires_grad


def test_save_attention_maps_writes_first_batch_rows_once_per_epoch(tmp_path):
    torch.manual_seed(seed=0)
    model = AGNewsClassifier(vocab_size=bpe.PAD_ID + 1)
    first = _padded_ids(batch_size=5, length=12)
    second = _padded_ids(batch_size=5, length=12)
    for epoch in (1, 3):
        for batch_index, ids in enumerate((first, second)):  # noqa: NAR001
            predicted = model(ids)  # noqa: NAR001
            save_attention_maps(
                model=model,
                input_tensor=ids,
                target_tensor=ids[:, 0],
                predicted=predicted,
                epoch=epoch,
                batch_index=batch_index,
                out_dir=tmp_path,
                num_sentences=2,
            )

    assert sorted(path.name for path in tmp_path.glob(pattern="*.pt")) == [  # noqa: NAR001
        "epoch_001.pt",
        "epoch_003.pt",
    ]
    capture = torch.load(f=tmp_path / "epoch_001.pt")
    assert torch.equal(capture["token_ids"], first[:2])  # noqa: NAR001
    assert len(capture["weights"]) == NUM_BLOCKS  # noqa: NAR001
    assert capture["weights"][0].shape == (2, NUM_HEADS, 12, 12)


def test_small_training_run_reduces_train_loss(tmp_path):
    # Reuse the cached rows and BPE merges instead of get_ag_news_dataloader, which
    # would retrain the tokenizer and encode every cached row.
    sampled_dir = DATASETS_CACHE_DIR / "ag_news" / "sampled"
    cached = _load_merges(path=sampled_dir / "train" / "bpe.json", num_merges=2000)
    if cached is None:
        pytest.skip(reason="needs the cached AG News splits and BPE merges")
    merges, vocab = cached
    train_rows = _read_split(path=sampled_dir / "train" / "rows.csv")[:500]
    eval_rows = _read_split(path=sampled_dir / "eval" / "rows.csv")[:100]
    train_loader = DataLoader(
        dataset=_encode_rows(rows=train_rows, merges=merges, max_len=MAX_LEN),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=_encode_rows(rows=eval_rows, merges=merges, max_len=MAX_LEN),
        batch_size=64,
    )
    torch.manual_seed(seed=0)
    model = AGNewsClassifier(vocab_size=len(vocab))
    state = TrainState(
        model=model,
        optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
    )
    epoch_spec = EpochSpec(
        device=torch.device("cpu"),
        metrics_type=ClassificationMetrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
    )

    history = fit(
        state=state,
        epoch_spec=epoch_spec,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        val_epoch_list=[1, 2, 3],
        history_path=tmp_path / "history.json",
    )

    assert history[-1].train_loss < history[0].train_loss


def test_save_attention_maps_rejects_model_without_attention_map(tmp_path):
    ids = _padded_ids(batch_size=2, length=4)
    with pytest.raises(TypeError):
        save_attention_maps(
            model=torch.nn.Linear(in_features=1, out_features=1),
            input_tensor=ids,
            target_tensor=ids[:, 0],
            predicted=ids.float(),
            epoch=1,
            batch_index=0,
            out_dir=tmp_path,
            num_sentences=1,
        )
