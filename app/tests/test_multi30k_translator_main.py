import pytest
import torch
from training.dataclass.config import (
    AttentionConfig,
    BlockConfig,
    EmbeddingConfig,
    TransformerConfig,
)
from training.model.transformer_translator import TransformerTranslator
from training.preprocess.constants import PROTECTED_IDS
from training.run.multi30k_translator.main import greedy_translate, translate_with_cache

SEQ_LEN = 8
VOCAB = 270


class _ByteTokenizer:
    """Bytes in, ids out; drops protected ids on the way back, as the real one does."""

    def encode(self, texts, cache=None):
        return [list(text.encode()) for text in texts]

    def decode(self, ids):
        return [" ".join(str(i) for i in row if i not in PROTECTED_IDS) for row in ids]


def _model(qk_norm: bool) -> TransformerTranslator:
    config = TransformerConfig(
        d_model=16,
        src=EmbeddingConfig(seq_length=SEQ_LEN, dropout=0.1),
        tgt=EmbeddingConfig(seq_length=SEQ_LEN, dropout=0.1),
        num_encoders=1,
        num_decoders=2,
        encoder=BlockConfig(attention=AttentionConfig(h=2)),
        decoder=BlockConfig(attention=AttentionConfig(h=2, qk_norm=qk_norm)),
    )
    return TransformerTranslator(
        model_config=config, src_vocab_size=VOCAB, tgt_vocab_size=VOCAB
    ).eval()


@pytest.mark.parametrize(argnames="qk_norm", argvalues=[False, True])
@pytest.mark.parametrize(argnames="sentence", argvalues=["a", "two dogs", "a long one!"])
def test_cached_translation_matches_uncached(qk_norm, sentence):
    torch.manual_seed(0)
    model = _model(qk_norm=qk_norm)
    kwargs = dict(
        src_sentence=sentence,
        transformer_model=model,
        device=torch.device("cpu"),
        src_tokenizer=_ByteTokenizer(),
        tgt_tokenizer=_ByteTokenizer(),
    )
    uncached = greedy_translate(**kwargs)
    assert uncached
    assert translate_with_cache(**kwargs) == uncached
