import pytest
import torch
from torch import nn
from training.dataclass.config import (
    AttentionConfig,
    BlockConfig,
    EmbeddingConfig,
    TransformerConfig,
    NormName,
)
from training.model.multi30k_translator import Multi30kTranslator

SEQ_LEN = 8
SRC_VOCAB, TGT_VOCAB = 300, 310


def _config(decoder_norm: NormName = "layernorm") -> TransformerConfig:
    return TransformerConfig(
        d_model=16,
        src=EmbeddingConfig(seq_length=SEQ_LEN, dropout=0.1),
        tgt=EmbeddingConfig(seq_length=SEQ_LEN, dropout=0.1),
        num_encoders=1,
        num_decoders=1,
        encoder=BlockConfig(attention=AttentionConfig(h=2)),
        decoder=BlockConfig(attention=AttentionConfig(h=2), norm=decoder_norm),
    )


def test_forward_returns_one_row_per_target_position():
    model = Multi30kTranslator(
        model_config=_config(), src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB
    ).eval()
    batch = torch.randint(low=0, high=SRC_VOCAB, size=(3, 2, SEQ_LEN))
    assert model(batch).shape == (3 * SEQ_LEN, TGT_VOCAB)


def test_each_stack_picks_its_own_norm():
    model = Multi30kTranslator(
        model_config=_config(decoder_norm="rmsnorm"),
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
    )
    assert type(model.decoder_norm) is nn.LayerNorm
    assert type(model.pred_norm) is nn.RMSNorm
    assert type(model.decoder_blocks[0].ffn_norm) is nn.RMSNorm
    assert type(model.encoder_blocks[0].ffn_norm) is nn.LayerNorm


def test_rejects_a_side_without_position_encoding():
    config = _config().with_overrides(overrides={"tgt": {"position_encoding": False}})
    with pytest.raises(expected_exception=ValueError, match="position encoding"):
        Multi30kTranslator(model_config=config, src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB)
