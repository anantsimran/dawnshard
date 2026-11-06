import json

import pytest
from training.preprocess import tokenizer as tokenizer_module
from training.preprocess.bpe import gpt4_preprocessor, train
from training.preprocess.tokenizer import get_tokenizer

TEXTS = ["low lower lowest newer newest wider widest"] * 3


@pytest.fixture
def trainings(monkeypatch):
    """Record every call to build_tokenizer with chunks, i.e. every training run."""
    calls = []
    real = train.build_tokenizer

    def spy(chunks=None, num_merges=None, saved_state=None):
        if chunks is not None:
            calls.append(num_merges)
        return real(chunks=chunks, num_merges=num_merges, saved_state=saved_state)

    monkeypatch.setattr(target=tokenizer_module.train, name="build_tokenizer", value=spy)
    return calls


def test_trains_and_writes_when_missing(tmp_path, trainings):
    path = tmp_path / "nested" / "t.json"
    tokenizer = get_tokenizer(merges_path=path, train_texts=TEXTS, num_merges=5)
    assert trainings == [5]
    assert json.loads(path.read_text(encoding="utf-8")) == [
        list(pair) for pair in tokenizer.state()
    ]


def test_trained_merges_match_build_tokenizer(tmp_path):
    tokenizer = get_tokenizer(merges_path=tmp_path / "t.json", train_texts=TEXTS, num_merges=5)
    direct = train.build_tokenizer(
        chunks=gpt4_preprocessor.preprocess(text="\n".join(TEXTS)), num_merges=5
    )
    assert tokenizer.state() == direct.state()


def test_loads_without_retraining_when_present(tmp_path, trainings):
    path = tmp_path / "t.json"
    first = get_tokenizer(merges_path=path, train_texts=TEXTS, num_merges=5)
    second = get_tokenizer(merges_path=path, train_texts=["unrelated"], num_merges=9)
    assert trainings == [5]
    assert second.cache_key() == first.cache_key()


def test_loads_without_train_texts(tmp_path, trainings):
    path = tmp_path / "t.json"
    first = get_tokenizer(merges_path=path, train_texts=TEXTS, num_merges=5)
    assert get_tokenizer(merges_path=path).cache_key() == first.cache_key()
    assert trainings == [5]


def test_missing_file_without_train_texts_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_tokenizer(merges_path=tmp_path / "t.json")


def test_no_num_merges_trains_the_default(tmp_path, trainings, monkeypatch):
    monkeypatch.setattr(target=train, name="DEFAULT_NUM_MERGES", value=4)
    tokenizer = get_tokenizer(merges_path=tmp_path / "t.json", train_texts=TEXTS)
    assert trainings == [None]
    assert len(tokenizer.state()) == 4


def test_invalid_merges_file_raises(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps([[97, 256]]), encoding="utf-8")  # 256 is PAD_ID
    with pytest.raises(ValueError):
        get_tokenizer(merges_path=path)


def test_returned_tokenizer_round_trips_text(tmp_path):
    tokenizer = get_tokenizer(merges_path=tmp_path / "t.json", train_texts=TEXTS, num_merges=8)
    assert tokenizer.decode(ids=tokenizer.encode(texts=["lowest\twider"])) == ["lowest wider"]
