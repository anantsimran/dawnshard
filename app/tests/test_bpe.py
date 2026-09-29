import pytest
from training.common import bpe
from training.common.bpe import Phrase
from training.common.constants import CLS_ID, MASK_ID, PAD_ID, PROTECTED_IDS, SEP_ID


def test_phrase_requires_encoding_xor_merge():
    with pytest.raises(ValueError):
        Phrase()
    with pytest.raises(ValueError):
        Phrase(encoding=65, first=Phrase(encoding=65), second=Phrase(encoding=66))


def test_phrase_to_str_leaf():
    assert Phrase(encoding=65).to_str(vocab={65: "A"}) == "A"


def test_phrase_to_str_merge_concatenates_children():
    merge = Phrase(first=Phrase(encoding=65), second=Phrase(encoding=66))
    assert merge.to_str(vocab={65: "A", 66: "B"}) == "AB"


def test_train_learns_most_frequent_pair():
    merges, vocab = bpe.train(text="aaaa bb", num_merges=1)
    assert len(merges) == 1
    assert vocab[bpe.FIRST_MERGE_ID] == "aa"


def test_train_vocab_includes_pad_id_so_len_is_embedding_size():
    merges, vocab = bpe.train(text="aaaa bb", num_merges=1)
    assert vocab[PAD_ID] == ""
    assert sorted(vocab) == list(range(bpe.FIRST_MERGE_ID + len(merges)))


def test_encode_never_emits_pad_id_and_numbers_merges_after_it():
    merges, _ = bpe.train(text="aaaa bb", num_merges=1)
    ids = bpe.encode(text="aaaa bb", merges=merges)
    assert PAD_ID not in ids
    assert ids[:2] == [bpe.FIRST_MERGE_ID, bpe.FIRST_MERGE_ID]


def test_decode_drops_pad_id():
    merges, vocab = bpe.train(text="ab", num_merges=0)
    assert bpe.decode(ids=[97, 98, PAD_ID], vocab=vocab) == "ab"


def test_protected_ids_sit_between_bytes_and_merges():
    assert PROTECTED_IDS == list(range(256, bpe.FIRST_MERGE_ID))


def test_train_vocab_names_protected_ids():
    _, vocab = bpe.train(text="ab", num_merges=0)
    assert [vocab[i] for i in (MASK_ID, CLS_ID, SEP_ID)] == ["[MASK]", "[CLS]", "[SEP]"]


def test_decode_drops_protected_ids():
    _, vocab = bpe.train(text="ab", num_merges=0)
    ids = [CLS_ID, 97, MASK_ID, SEP_ID, 98, PAD_ID]
    assert bpe.decode(ids=ids, vocab=vocab) == "ab"


def test_train_keeps_merging_while_a_single_distinct_pair_repeats():
    merges, vocab = bpe.train(text="aaaaaaaa", num_merges=10)
    assert [vocab[bpe.FIRST_MERGE_ID + k] for k in range(len(merges))] == ["aa", "aaaa"]


def test_train_never_merges_across_chunks():
    # Chunks are 'The', ' dog', '.', ' The', ' dog', '!': "g." and "g!" never form.
    merges, vocab = bpe.train(text="The dog. The dog!", num_merges=50)
    learned = [vocab[bpe.FIRST_MERGE_ID + k] for k in range(len(merges))]
    assert " dog" in learned
    assert not any("." in token or "!" in token for token in learned)


def test_train_no_merges_when_no_pair_repeats():
    merges, _ = bpe.train(text="abcd", num_merges=5)
    assert merges == []


def test_encode_merges_overlapping_runs_left_to_right():
    merges, _ = bpe.train(text="aaaa bb", num_merges=1)
    assert bpe.encode(text="aaa", merges=merges) == [bpe.FIRST_MERGE_ID, 97]


def test_encode_with_no_merges_returns_raw_bytes():
    assert bpe.encode(text="AB", merges=[]) == [65, 66]


def test_encode_decode_roundtrip_is_lossless_for_plain_ascii():
    text = "the cat sat on the mat"
    merges, vocab = bpe.train(text=text, num_merges=50)
    ids = bpe.encode(text=text, merges=merges)
    assert bpe.decode(ids=ids, vocab=vocab) == text


def test_decode_normalizes_whitespace_to_a_single_space():
    text = "a\tb\nc"
    merges, vocab = bpe.train(text=text, num_merges=10)
    ids = bpe.encode(text=text, merges=merges)
    assert bpe.decode(ids=ids, vocab=vocab) == "a b c"


def test_decode_replaces_invalid_utf8_with_replacement_char():
    assert bpe.decode(ids=[255], vocab={255: chr(255)}) == "�"
