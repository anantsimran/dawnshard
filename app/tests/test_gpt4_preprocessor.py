from training.preprocess.bpe.gpt4_preprocessor import preprocess, to_text


def test_preprocess_splits_words_punctuation_and_digit_runs():
    assert preprocess(text="The dog.\tIt ran 12345 m") == [
        b"The",
        b" dog",
        b".",
        b" It",
        b" ran",
        b" ",
        b"123",
        b"45",
        b" m",
    ]


def test_preprocess_maps_every_whitespace_char_to_one_space():
    assert b"".join(preprocess(text="a\tb\nc d")) == b"a b c d"


def test_preprocess_returns_utf8_bytes():
    assert preprocess(text="é") == ["é".encode("utf-8")]


def test_preprocess_empty_text_has_no_chunks():
    assert preprocess(text="") == []


def test_to_text_inverts_preprocess_up_to_whitespace():
    assert to_text(chunks=b"".join(preprocess(text="Männer\tran,\n12345 m"))) == (
        "Männer ran, 12345 m"
    )


def test_to_text_replaces_invalid_utf8_with_replacement_char():
    assert to_text(chunks=b"a\xffb") == "a\ufffdb"
