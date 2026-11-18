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
from training.model.transformer_translator import TransformerTranslator
from training.preprocess.constants import PAD_ID
from training.transformer.masks import mask_later_positions, mask_tokenizer_pad_ids

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
    model = TransformerTranslator(
        model_config=_config(), src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB
    ).eval()
    batch = torch.randint(low=0, high=SRC_VOCAB, size=(3, 2, SEQ_LEN))
    assert model(batch).shape == (3 * SEQ_LEN, TGT_VOCAB)
    assert {kind: len(blocks) for kind, blocks in model.attention_maps.map.items()} == {
        "encoder": 1,
        "decoder_self": 1,
        "decoder_cross": 1,
    }


def test_each_stack_picks_its_own_norm():
    model = TransformerTranslator(
        model_config=_config(decoder_norm="rmsnorm"),
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
    )
    assert type(model.encoder_norm) is nn.LayerNorm
    assert type(model.pred_norm) is nn.RMSNorm
    assert type(model.decoder_blocks[0].ffn_norm) is nn.RMSNorm
    assert type(model.encoder_blocks[0].ffn_norm) is nn.LayerNorm


def test_rejects_a_side_without_position_encoding():
    config = _config().with_overrides(overrides={"tgt": {"position_encoding": False}})
    with pytest.raises(expected_exception=ValueError, match="position encoding"):
        TransformerTranslator(model_config=config, src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB)


def test_decode_next_token_matches_teacher_forcing_at_each_prefix():
    config = _config().with_overrides(overrides={"num_decoders": 2})
    model = TransformerTranslator(
        model_config=config, src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB
    ).eval()
    # Ids below PAD_ID, so neither side has padding.
    batch = torch.randint(low=0, high=PAD_ID, size=(3, 2, SEQ_LEN))
    src, tgt_in = batch[:, 0], batch[:, 1]
    full = model(batch).reshape(3, SEQ_LEN, TGT_VOCAB)
    encoder_mask = mask_tokenizer_pad_ids(token_ids=src, pad_id=PAD_ID)
    memory, _ = model.encode(src=src, encoder_mask=encoder_mask)
    for t in range(1, SEQ_LEN + 1):
        step = model.decode_next_token(
            tgt_in=tgt_in[:, :t],
            memory=memory,
            decoder_mask=mask_later_positions(seq_len=t),
            memory_mask=encoder_mask,
        )
        torch.testing.assert_close(actual=step, expected=full[:, t - 1])
