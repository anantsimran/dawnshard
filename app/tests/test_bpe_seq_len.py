from training.preprocess.bpe import gpt4_preprocessor, train
from training.viz.bpe_seq_len import merge_counts


def test_merge_counts_never_merges_across_chunks():
    # "1a" is the chunks '1' and 'a', so the pair (49, 97) never occurs in one chunk.
    assert merge_counts(texts=["1a"], pairs=[[49, 97]]).tolist() == [0]


def test_merge_counts_match_token_counts_from_encode():
    corpus = "the cat sat on the mat. 12345 cats, 678 mats! the cat's hat"
    tokenizer = train.build_tokenizer(
        chunks=gpt4_preprocessor.preprocess(text=corpus), num_merges=30
    )
    pairs = [list(pair) for pair in tokenizer.state()]
    texts = ["the cat sat", "cats on 12345 mats", "the hat's mat."]
    counts = merge_counts(texts=texts, pairs=pairs)
    total_bytes = sum(len(chunk) for t in texts for chunk in gpt4_preprocessor.preprocess(text=t))
    for k in range(len(pairs) + 1):
        prefix = train.build_tokenizer(saved_state=tokenizer.state()[:k])
        encoded = prefix.encode(texts=texts)
        expected = sum(len(ids) for ids in encoded)
        assert total_bytes - counts[:k].sum() == expected
