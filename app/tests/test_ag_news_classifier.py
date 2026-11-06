import csv
import io

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from training.constants import TEST_SEED, VAL_SEED
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataload import ag_news
from training.dataload.ag_news import (
    AGNewsDataset,
    _encode_rows,
    _get_raw_rows,
    get_ag_news_texts,
)
from training.metrics.classification_metrics import (
    ClassificationMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.model.ag_news_classifier import AGNewsClassifier
from training.preprocess.bpe.gpt4_preprocessor import preprocess
from training.preprocess.bpe.train import build_tokenizer
from training.preprocess.constants import PAD_ID
from training.dataclass.config import FitConfig
from training.dataclass.model import EpochSpec, TrainState
from training.run.ag_news_classifier import DEFAULT_CONFIG_PATH, AGNewsClassifierTrainConfig
from training.train.train_loop import fit
from training.transformer.probes import save_attention_maps


MODEL_CONFIG = AGNewsClassifierTrainConfig.from_yaml(path=DEFAULT_CONFIG_PATH).model
MAX_LEN = MODEL_CONFIG.embedding.seq_length
NUM_BLOCKS = MODEL_CONFIG.num_encoders
NUM_HEADS = MODEL_CONFIG.encoder.attention.h


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_csv_download(monkeypatch, rows):
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)  # noqa: NAR001
    monkeypatch.setattr(
        target=ag_news.urllib.request,
        name="urlopen",
        value=lambda url: _FakeResponse(buffer.getvalue().encode("utf-8")),
    )


def test_get_raw_rows_converts_label_to_zero_indexed_and_joins_title_and_description(
    monkeypatch,
):
    _fake_csv_download(monkeypatch=monkeypatch, rows=[["1", "Title", "Description"]])

    assert _get_raw_rows(split="train") == [("Title Description", 0)]


def test_get_raw_rows_replaces_escaped_linebreaks_with_space(monkeypatch):
    _fake_csv_download(monkeypatch=monkeypatch, rows=[["2", "Title", r"line1\line2"]])

    assert _get_raw_rows(split="train") == [("Title line1 line2", 1)]


def test_get_raw_rows_keeps_linebreaks_inside_quoted_fields(monkeypatch):
    _fake_csv_download(monkeypatch=monkeypatch, rows=[["3", "Title", "line1\nline2"]])

    assert _get_raw_rows(split="train") == [("Title line1\nline2", 2)]


def test_get_raw_rows_downloads_once_with_a_cache(monkeypatch, tmp_path):
    calls = []
    _fake_csv_download(monkeypatch=monkeypatch, rows=[["1", "T", "D"]])
    fake = ag_news.urllib.request.urlopen
    monkeypatch.setattr(
        target=ag_news.urllib.request,
        name="urlopen",
        value=lambda url: calls.append(url) or fake(url=url),
    )
    with SqliteDao(path=tmp_path / "c.sqlite") as cache:
        _get_raw_rows(split="train", cache=cache)
        _get_raw_rows(split="train", cache=cache)
    assert len(calls) == 1


def _labelled_rows(count, label_offset=0):
    return [(f"row {i}", (i + label_offset) % 4) for i in range(count)]


def test_get_ag_news_texts_samples_disjoint_eval_and_test_and_is_seeded(monkeypatch):
    raw = {"train": _labelled_rows(50), "test": _labelled_rows(40)}
    monkeypatch.setattr(
        target=ag_news,
        name="_get_raw_rows",
        value=lambda split, cache=None: raw[split],
    )

    first = get_ag_news_texts(n_train=10, n_eval=5, n_test=6, seed=1)
    again = get_ag_news_texts(n_train=10, n_eval=5, n_test=6, seed=1)

    assert first == again
    assert [len(first[split]) for split in ("train", "eval", "test")] == [10, 5, 6]
    assert not set(first["eval"]) & set(first["test"])
    assert set(first["train"]) <= set(raw["train"])
    assert set(first["eval"] + first["test"]) <= set(raw["test"])


def test_get_ag_news_texts_keys_the_cache_by_sizes_and_seed(monkeypatch, tmp_path):
    calls = []

    def fake_raw_rows(split, cache=None):
        calls.append(split)
        return _labelled_rows(40)

    monkeypatch.setattr(target=ag_news, name="_get_raw_rows", value=fake_raw_rows)
    with SqliteDao(path=tmp_path / "c.sqlite") as cache:
        get_ag_news_texts(n_train=5, n_eval=2, n_test=2, seed=0, cache=cache)
        get_ag_news_texts(n_train=5, n_eval=2, n_test=2, seed=0, cache=cache)
        assert len(calls) == 2
        get_ag_news_texts(n_train=6, n_eval=2, n_test=2, seed=0, cache=cache)
        assert len(calls) == 4
        get_ag_news_texts(n_train=5, n_eval=2, n_test=2, seed=1, cache=cache)
        assert len(calls) == 6


def test_encode_rows_pads_to_max_len_with_pad_id():
    dataset = _encode_rows(encoded=[[97, 98]], labels=[0], max_len=5)

    assert isinstance(dataset, AGNewsDataset)
    assert len(dataset) == 1
    ids, label = dataset[0]
    pad = PAD_ID
    assert torch.equal(ids, torch.tensor([97, 98, pad, pad, pad]))
    assert label == 0


def test_encode_rows_truncates_to_max_len():
    dataset = _encode_rows(encoded=[[97, 98, 99, 100]], labels=[0], max_len=2)

    ids, _ = dataset[0]
    assert torch.equal(ids, torch.tensor([97, 98]))


def _padded_ids(batch_size, length):
    ids = torch.randint(low=0, high=PAD_ID, size=(batch_size, length))
    ids[:, -2:] = PAD_ID
    return ids


def test_classifier_forward_stores_detached_attention_map_per_block():
    torch.manual_seed(seed=0)
    model = AGNewsClassifier(model_config=MODEL_CONFIG, vocab_size=PAD_ID + 1)
    model(_padded_ids(batch_size=3, length=12))  # noqa: NAR001
    assert list(model.attention_map) == ["encoder"]  # noqa: NAR001
    assert len(model.attention_map["encoder"]) == NUM_BLOCKS  # noqa: NAR001
    for weights in model.attention_map["encoder"]:
        assert weights.shape == (3, NUM_HEADS, 12, 12)
        assert not weights.requires_grad


def test_classifier_is_sensitive_to_token_order():
    torch.manual_seed(seed=0)
    model = AGNewsClassifier(model_config=MODEL_CONFIG, vocab_size=PAD_ID + 1).eval()
    ids = _padded_ids(batch_size=3, length=12)
    reversed_ids = ids.clone()
    reversed_ids[:, :-2] = ids[:, :-2].flip(dims=(1,))
    with torch.no_grad():
        logits = model(ids)  # noqa: NAR001
        reversed_logits = model(reversed_ids)  # noqa: NAR001
    assert not torch.allclose(logits, reversed_logits, atol=1e-5)  # noqa: NAR001


def test_save_attention_maps_writes_first_batch_rows_once_per_epoch(tmp_path):
    torch.manual_seed(seed=0)
    model = AGNewsClassifier(model_config=MODEL_CONFIG, vocab_size=PAD_ID + 1)
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
                seed=VAL_SEED,
                out_dir=tmp_path,
                num_sentences=2,
            )

    assert sorted(path.name for path in tmp_path.glob(pattern="*.pt")) == [  # noqa: NAR001
        "epoch_001.pt",
        "epoch_003.pt",
    ]
    capture = torch.load(f=tmp_path / "epoch_001.pt")
    assert torch.equal(capture["token_ids"], first[:2])  # noqa: NAR001
    assert len(capture["weights"]["encoder"]) == NUM_BLOCKS  # noqa: NAR001
    assert capture["weights"]["encoder"][0].shape == (2, NUM_HEADS, 12, 12)


def test_small_training_run_reduces_train_loss():
    # Each class has its own words, so a small model can learn them in three epochs.
    class_words = ["senate vote", "goal match", "stock market", "chip software"]
    rows = [(f"{class_words[i % 4]} story {i}", i % 4) for i in range(600)]
    tokenizer = build_tokenizer(
        chunks=preprocess(text="\n".join(text for text, _ in rows)), num_merges=50
    )
    train_rows, eval_rows = rows[:500], rows[500:]

    def loader(subset, shuffle):
        return DataLoader(
            dataset=_encode_rows(
                encoded=tokenizer.encode(texts=[text for text, _ in subset]),
                labels=[label for _, label in subset],
                max_len=MAX_LEN,
            ),
            batch_size=64,
            shuffle=shuffle,
        )

    train_loader = loader(subset=train_rows, shuffle=True)
    val_loader = loader(subset=eval_rows, shuffle=False)
    torch.manual_seed(seed=0)
    model = AGNewsClassifier(model_config=MODEL_CONFIG, vocab_size=len(tokenizer.vocab))
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
        fit_config=FitConfig(
            num_epochs=3,
            val_epoch_list=[1, 2, 3],
            train_seed=0,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
        ),
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
            seed=VAL_SEED,
            out_dir=tmp_path,
            num_sentences=1,
        )
