from training.preprocess.bpe.decode import decode_ids
from training.preprocess.bpe.encode import encode_chunks
from training.preprocess.bpe.gpt4_preprocessor import preprocess
from training.preprocess.bpe.tokenizer import BpeTokenizer
from training.preprocess.bpe.train import build_tokenizer
from training.preprocess.constants import CLS_ID, MASK_ID, PAD_ID, SEP_ID


def test_decode_drops_pad_id():
    assert decode_ids(tokenizer=BpeTokenizer(merges=[]), ids=[[97, 98, PAD_ID]]) == [b"ab"]


def test_decode_drops_protected_ids():
    ids = [CLS_ID, 97, MASK_ID, SEP_ID, 98, PAD_ID]
    assert decode_ids(tokenizer=BpeTokenizer(merges=[]), ids=[ids]) == [b"ab"]


def test_decode_returns_the_bytes_of_the_encoded_chunks():
    text = "the cat sat on the mat, café"
    tokenizer = build_tokenizer(chunks=preprocess(text=text), num_merges=50)
    ids = encode_chunks(tokenizer=tokenizer, chunks=[preprocess(text=text)])
    assert decode_ids(tokenizer=tokenizer, ids=ids) == [b"".join(preprocess(text=text))]


def test_decode_returns_bytes_even_when_they_are_not_valid_utf8():
    assert decode_ids(tokenizer=BpeTokenizer(merges=[]), ids=[[255]]) == [b"\xff"]
