import json
import random
from collections import Counter

import pytest
from loguru import logger
from training.preprocess.bpe import train
from training.preprocess.bpe.encode import encode_chunk
from training.preprocess.bpe.gpt4_preprocessor import preprocess
from training.preprocess.bpe.train import build_tokenizer
from training.preprocess.constants import FIRST_MERGE_ID, PAD_ID


def _train(text, num_merges):
    return build_tokenizer(chunks=preprocess(text=text), num_merges=num_merges)


def _learned(tokenizer):
    return [tokenizer.vocab[FIRST_MERGE_ID + k] for k in range(len(tokenizer.merges))]


def test_build_learns_most_frequent_pair():
    tokenizer = _train(text="aaaa bb", num_merges=1)
    assert len(tokenizer.merges) == 1
    assert tokenizer.vocab[FIRST_MERGE_ID] == "aa"


def test_build_defaults_to_default_num_merges(monkeypatch):
    monkeypatch.setattr(target=train, name="DEFAULT_NUM_MERGES", value=2)
    tokenizer = build_tokenizer(chunks=preprocess(text="aaaaaaaa bbbb"))
    assert len(tokenizer.merges) == 2


def test_build_keeps_merging_while_a_single_distinct_pair_repeats():
    assert _learned(tokenizer=_train(text="aaaaaaaa", num_merges=10)) == ["aa", "aaaa"]


def test_build_never_merges_across_chunks():
    # Chunks are 'The', ' dog', '.', ' The', ' dog', '!': "g." and "g!" never form.
    learned = _learned(tokenizer=_train(text="The dog. The dog!", num_merges=50))
    assert " dog" in learned
    assert not any("." in token or "!" in token for token in learned)


def test_build_no_merges_when_no_pair_repeats():
    assert _train(text="abcd", num_merges=5).merges == []


def test_state_roundtrips_through_json_and_saved_state():
    tokenizer = _train(text="the cat sat on the mat", num_merges=20)
    saved = json.loads(json.dumps(tokenizer.state()))  # pairs come back as lists
    loaded = build_tokenizer(saved_state=saved)
    assert loaded.state() == tokenizer.state()
    assert loaded.vocab == tokenizer.vocab
    assert loaded.merges == tokenizer.merges


def test_build_from_saved_state_and_chunks_continues_to_num_merges_total():
    text = "low lower lowest newer newest wider widest"
    full = _train(text=text, num_merges=12)
    resumed = build_tokenizer(
        chunks=preprocess(text=text), num_merges=12, saved_state=full.state()[:4]
    )
    assert resumed.state()[:4] == full.state()[:4]
    assert len(resumed.merges) == 12
    assert len(set(resumed.state())) == 12  # no saved merge is learned again


def test_build_with_more_saved_merges_than_num_merges_warns_and_keeps_them():
    saved = _train(text="aaaaaaaa bbbb", num_merges=3).state()
    messages = []
    handler = logger.add(sink=messages.append, level="WARNING")
    try:
        tokenizer = build_tokenizer(chunks=preprocess(text="ab"), num_merges=1, saved_state=saved)
    finally:
        logger.remove(handler_id=handler)
    assert tokenizer.state() == saved
    assert len(messages) == 1


def test_build_rejects_no_arguments():
    with pytest.raises(ValueError):
        build_tokenizer()


def test_build_rejects_num_merges_without_chunks():
    with pytest.raises(ValueError):
        build_tokenizer(num_merges=4, saved_state=[(97, 98)])


def test_build_rejects_negative_num_merges():
    with pytest.raises(ValueError):
        build_tokenizer(chunks=[b"ab"], num_merges=-1)


def test_build_rejects_text_passed_as_chunks():
    with pytest.raises(TypeError):
        build_tokenizer(chunks="low lower lowest", num_merges=4)


@pytest.mark.parametrize(
    "saved_state",
    [
        [(97, PAD_ID)],  # protected id
        [(97, FIRST_MERGE_ID)],  # a merge that is not earlier
        [(97,)],  # not a pair
    ],
)
def test_build_rejects_invalid_saved_state(saved_state):
    with pytest.raises(ValueError):
        build_tokenizer(saved_state=saved_state)


def _naive_merges(chunks, num_merges, saved_state=()):
    """Recount every pair on every step; ties go to the smallest (first_id, second_id)."""
    saved = build_tokenizer(saved_state=list(saved_state))
    sequences = [encode_chunk(tokenizer=saved, chunk=chunk) for chunk in chunks]
    learned = list(saved_state)
    while len(learned) < num_merges:
        counts = Counter(pair for ids in sequences for pair in zip(ids, ids[1:]))
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        if counts[best] < 2:
            break
        new_id = FIRST_MERGE_ID + len(learned)
        learned.append(best)
        merged_sequences = []
        for ids in sequences:
            merged, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            merged_sequences.append(merged)
        sequences = merged_sequences
    return learned


@pytest.mark.parametrize(
    "text",
    [
        "low lower lowest",
        "aaaa aaa aa aaaaaaa a aaaa abab ababab",
        "ab cd ab cd ef gh ef gh",
        "café café naïve naïve 12345 12345 === ===",
    ]
    + ["".join(random.Random(seed).choices(population="ab c\n", k=120)) for seed in range(20)],
)
def test_build_matches_recounting_every_pair_on_every_step(text):
    chunks = preprocess(text=text)
    tokenizer = build_tokenizer(chunks=chunks, num_merges=40)
    assert tokenizer.state() == _naive_merges(chunks=chunks, num_merges=40)


def test_build_continuing_from_saved_state_matches_recounting():
    saved = _train(text="the cat sat on the mat the cat", num_merges=3).state()
    chunks = preprocess(text="the cat sat on the mat the cat that sat the hat")
    tokenizer = build_tokenizer(chunks=chunks, num_merges=12, saved_state=saved)
    assert tokenizer.state() == _naive_merges(chunks=chunks, num_merges=12, saved_state=saved)


def test_build_breaks_count_ties_by_the_smallest_pair():
    # "ab" and "cd" both occur twice; (97, 98) is smaller than (99, 100). Chunks are
    # passed directly, since preprocess would attach a space and add (32, 97).
    tokenizer = build_tokenizer(chunks=[b"cd", b"ab", b"cd", b"ab"], num_merges=1)
    assert tokenizer.state() == [(97, 98)]


def test_build_ignores_chunk_order():
    chunks = preprocess(text="the cat sat on the mat the cat that sat the hat")
    forward = build_tokenizer(chunks=chunks, num_merges=10).state()
    assert build_tokenizer(chunks=chunks[::-1], num_merges=10).state() == forward


def test_build_continuing_from_its_own_prefix_matches_one_straight_run():
    chunks = preprocess(text="the cat sat on the mat the cat that sat the hat " * 3)
    straight = build_tokenizer(chunks=chunks, num_merges=15).state()
    continued = build_tokenizer(chunks=chunks, num_merges=15, saved_state=straight[:5])
    assert continued.state() == straight
