import pytest
from training.dal.dao.sqlite_dao import SqliteDao
from training.preprocess.bpe import tokenizer as bpe_tokenizer
from training.preprocess.bpe.encode import encode_chunks
from training.preprocess.bpe.gpt4_preprocessor import preprocess
from training.preprocess.bpe.tokenizer import BpeTokenizer
from training.preprocess.bpe.train import build_tokenizer
from training.preprocess.constants import (
    CLS_ID,
    FIRST_MERGE_ID,
    MASK_ID,
    PAD_ID,
    PROTECTED_IDS,
    SEP_ID,
)


def _train(text, num_merges):
    return build_tokenizer(chunks=preprocess(text=text), num_merges=num_merges)


def test_version_is_a_dotted_string():
    assert all(part.isdigit() for part in bpe_tokenizer.__version__.split("."))


def test_vocab_includes_pad_id_so_len_is_embedding_size():
    tokenizer = _train(text="aaaa bb", num_merges=1)
    assert tokenizer.vocab[PAD_ID] == ""
    assert sorted(tokenizer.vocab) == list(range(FIRST_MERGE_ID + len(tokenizer.merges)))


def test_protected_ids_sit_between_bytes_and_merges():
    assert PROTECTED_IDS == list(range(256, FIRST_MERGE_ID))


def test_vocab_names_protected_ids():
    tokenizer = BpeTokenizer(merges=[])
    assert [tokenizer.vocab[i] for i in (MASK_ID, CLS_ID, SEP_ID)] == ["[MASK]", "[CLS]", "[SEP]"]


def test_encode_preprocesses_texts_and_matches_encode_chunks():
    tokenizer = _train(text="low lower lowest", num_merges=4)
    texts = ["lowest\tlow", "", "unseen ü"]
    assert tokenizer.encode(texts=texts) == encode_chunks(
        tokenizer=tokenizer, chunks=[preprocess(text=t) for t in texts]
    )


def test_encode_rejects_a_single_text():
    with pytest.raises(TypeError):
        BpeTokenizer(merges=[]).encode(texts="ab")


def test_encode_with_a_cache_matches_encode_without_one(tmp_path):
    tokenizer = _train(text="low lower lowest", num_merges=4)
    texts = ["lowest low", "low"]
    with SqliteDao(path=tmp_path / "c.sqlite") as cache:
        assert tokenizer.encode(texts=texts, cache=cache) == tokenizer.encode(texts=texts)
        assert tokenizer.encode(texts=texts, cache=cache) == tokenizer.encode(texts=texts)


def test_decode_returns_text_with_whitespace_as_spaces():
    tokenizer = _train(text="low lower lowest café", num_merges=10)
    ids = tokenizer.encode(texts=["lowest\tlow café"])
    assert tokenizer.decode(ids=[[CLS_ID, *ids[0], PAD_ID]]) == ["lowest low café"]


def test_tokenizer_id_is_stable_and_follows_merges_and_version(monkeypatch):
    first = _train(text="low lower lowest", num_merges=4)
    again = _train(text="low lower lowest", num_merges=4)
    fewer = _train(text="low lower lowest", num_merges=3)
    assert first.tokenizer_id == again.tokenizer_id == first.cache_key()
    assert first.tokenizer_id != fewer.tokenizer_id

    monkeypatch.setattr(target=bpe_tokenizer, name="__version__", value="0.0.0")
    assert BpeTokenizer(merges=first.merges).tokenizer_id != first.tokenizer_id
