import pytest
from training.dal.dao.sqlite_dao import SqliteDao
from training.preprocess.bpe.encode import encode_chunks
from training.preprocess.bpe.gpt4_preprocessor import preprocess
from training.preprocess.bpe.tokenizer import BpeTokenizer
from training.preprocess.bpe.train import build_tokenizer
from training.preprocess.constants import FIRST_MERGE_ID, PAD_ID


def _train(text, num_merges):
    return build_tokenizer(chunks=preprocess(text=text), num_merges=num_merges)


def _encode(tokenizer, texts):
    return encode_chunks(tokenizer=tokenizer, chunks=[preprocess(text=text) for text in texts])


def test_encode_never_emits_pad_id_and_numbers_merges_after_it():
    tokenizer = _train(text="aaaa bb", num_merges=1)
    [ids] = _encode(tokenizer=tokenizer, texts=["aaaa bb"])
    assert PAD_ID not in ids
    assert ids[:2] == [FIRST_MERGE_ID, FIRST_MERGE_ID]


def test_encode_returns_one_id_list_per_text_in_order():
    tokenizer = _train(text="low lower lowest newer newest", num_merges=20)
    texts = ["lowest newer", "", "low  low\tlower", "unseen bytes ü"]
    assert _encode(tokenizer=tokenizer, texts=texts) == [
        _encode(tokenizer=tokenizer, texts=[t])[0] for t in texts
    ]


def test_encode_accepts_a_generator():
    tokenizer = _train(text="ab ab", num_merges=1)
    chunks = (preprocess(text=t) for t in ["ab", "b"])
    assert encode_chunks(tokenizer=tokenizer, chunks=chunks) == [[FIRST_MERGE_ID], [98]]


def test_encode_rejects_one_texts_chunks_without_a_list_around_them():
    tokenizer = _train(text="ab", num_merges=0)
    with pytest.raises(TypeError):
        encode_chunks(tokenizer=tokenizer, chunks=preprocess(text="ab"))
    with pytest.raises(TypeError):
        encode_chunks(tokenizer=tokenizer, chunks=["ab"])


def test_encode_merges_overlapping_runs_left_to_right():
    tokenizer = _train(text="aaaa bb", num_merges=1)
    assert _encode(tokenizer=tokenizer, texts=["aaa"]) == [[FIRST_MERGE_ID, 97]]


def test_encode_with_no_merges_returns_raw_bytes():
    assert encode_chunks(tokenizer=BpeTokenizer(merges=[]), chunks=[[b"AB"]]) == [[65, 66]]


def test_encode_with_a_cache_matches_encode_without_one(tmp_path):
    tokenizer = _train(text="low lower lowest newer newest", num_merges=20)
    # Reordered, duplicated, and empty texts all come back in input order.
    texts = ["lowest newer", "low", "lowest newer", "", "unseen ü", "low"]
    chunks = [preprocess(text=text) for text in texts]
    expected = encode_chunks(tokenizer=tokenizer, chunks=chunks)
    with SqliteDao(path=tmp_path / "c.sqlite") as cache:
        assert encode_chunks(tokenizer=tokenizer, chunks=chunks, cache=cache) == expected
        assert encode_chunks(tokenizer=tokenizer, chunks=chunks, cache=cache) == expected
        reordered = list(reversed(chunks))
        assert encode_chunks(tokenizer=tokenizer, chunks=reordered, cache=cache) == list(
            reversed(expected)
        )


def test_encode_hits_the_cache_for_reordered_input(tmp_path):
    chunks = [preprocess(text=text) for text in ["low", "lower", "lowest"]]
    tokenizer = _train(text="low lower lowest", num_merges=4)
    with SqliteDao(path=tmp_path / "c.sqlite") as cache:
        encode_chunks(tokenizer=tokenizer, chunks=chunks, cache=cache)
        encode_chunks(tokenizer=tokenizer, chunks=chunks[::-1] + chunks, cache=cache)
        rows = cache._connection.execute("SELECT COUNT(*) FROM cache").fetchone()  # noqa: NAR001
    assert rows == (1,)


def test_encode_empty_input_returns_empty_list_without_touching_the_cache(tmp_path):
    tokenizer = _train(text="low lower lowest", num_merges=4)
    with SqliteDao(path=tmp_path / "c.sqlite") as cache:
        assert encode_chunks(tokenizer=tokenizer, chunks=[], cache=cache) == []
        rows = cache._connection.execute("SELECT COUNT(*) FROM cache").fetchone()  # noqa: NAR001
    assert rows == (0,)


def test_encode_with_duplicates_returns_independent_lists():
    tokenizer = _train(text="low lower lowest", num_merges=4)
    first, second = encode_chunks(tokenizer=tokenizer, chunks=[preprocess(text="low")] * 2)
    first.append(0)
    assert second != first
