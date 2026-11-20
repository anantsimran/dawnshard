from typing import cast

import pytest
import torch
from torch import nn
from training.dataclass.config import (
    AttentionConfig,
    BlockConfig,
    EmbeddingConfig,
    NormName,
    TransformerConfig,
)
from training.model.transformer_translator import TransformerTranslator
from training.preprocess.constants import PAD_ID
from training.transformer.masks import mask_later_positions, mask_tokenizer_pad_ids
from training.transformer.modules import DecoderBlock, DecoderBlockCache, KVCache

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
        TransformerTranslator(
            model_config=config, src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB
        )


def _two_block_model(qk_norm: bool = False) -> TransformerTranslator:
    config = _config().with_overrides(
        overrides={
            "num_decoders": 2,
            "decoder": {"attention": {"qk_norm": qk_norm}},
        }
    )
    return TransformerTranslator(
        model_config=config, src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB
    ).eval()


def _empty_cache(model: TransformerTranslator, memory: torch.Tensor) -> list[DecoderBlockCache]:
    # Room for every target position in self-attention, exactly S rows in cross.
    attn = cast(DecoderBlock, model.decoder_blocks[0]).self_attn

    def kv(rows: int) -> KVCache:
        shape = (memory.shape[0], attn.h, rows, attn.d_k)
        return KVCache(k=memory.new_empty(size=shape), v=memory.new_empty(size=shape))

    return [
        DecoderBlockCache(self_attn=kv(rows=SEQ_LEN), cross_attn=kv(rows=memory.shape[1]))
        for _ in model.decoder_blocks
    ]


def _teacher_forced(model, src_len=SEQ_LEN):
    # Ids below PAD_ID, so neither side has padding.
    src = torch.randint(low=0, high=PAD_ID, size=(3, src_len))
    tgt_in = torch.randint(low=0, high=PAD_ID, size=(3, SEQ_LEN))
    encoder_mask = mask_tokenizer_pad_ids(token_ids=src, pad_id=PAD_ID)
    memory, _ = model.encode(src=src, encoder_mask=encoder_mask)
    full, _, _ = model.decode(
        tgt_in=tgt_in,
        pos_encoding=model.positional_embedding.pe[:SEQ_LEN],
        memory=memory,
        decoder_mask=mask_later_positions(seq_len=SEQ_LEN),
        memory_mask=encoder_mask,
    )
    return tgt_in, memory, encoder_mask, full


def test_decode_of_each_prefix_matches_teacher_forcing():
    model = _two_block_model()
    tgt_in, memory, encoder_mask, full = _teacher_forced(model=model)
    causal = mask_later_positions(seq_len=SEQ_LEN)
    for t in range(1, SEQ_LEN + 1):
        step, _, _ = model.decode(
            tgt_in=tgt_in[:, :t],
            pos_encoding=model.positional_embedding.pe[:t],
            memory=memory,
            decoder_mask=causal[:, :, :t, :t],
            memory_mask=encoder_mask,
        )
        torch.testing.assert_close(actual=step[:, -1], expected=full[:, t - 1])


@pytest.mark.parametrize(argnames="qk_norm", argvalues=[False, True])
def test_cached_decode_one_token_at_a_time_matches_teacher_forcing(qk_norm):
    model = _two_block_model(qk_norm=qk_norm)
    # A source length unlike the target's, so a swapped self and cross cache fails.
    tgt_in, memory, encoder_mask, full = _teacher_forced(model=model, src_len=5)
    cache = _empty_cache(model=model, memory=memory)
    for t in range(SEQ_LEN):
        step, self_attn, _ = model.decode(
            tgt_in=tgt_in[:, t : t + 1],
            pos_encoding=model.positional_embedding.at_pos(pos=t).unsqueeze(dim=0),
            memory=memory,
            decoder_mask=None,
            memory_mask=encoder_mask,
            decoder_cache=cache,
        )
        torch.testing.assert_close(actual=step[:, 0], expected=full[:, t])
        assert self_attn[0].shape == (3, 2, 1, t + 1)
    assert [c.self_attn.length_filled for c in cache] == [SEQ_LEN, SEQ_LEN]
    assert [c.cross_attn.length_filled for c in cache] == [5, 5]


def test_decode_rejects_a_cache_per_block_mismatch():
    model = _two_block_model()
    _, memory, encoder_mask, _ = _teacher_forced(model=model)
    cache = _empty_cache(model=model, memory=memory)[:1]
    with pytest.raises(expected_exception=ValueError, match="shorter"):
        model.decode(
            tgt_in=torch.zeros(3, 1, dtype=torch.int64),
            pos_encoding=model.positional_embedding.pe[:1],
            memory=memory,
            decoder_mask=None,
            memory_mask=encoder_mask,
            decoder_cache=cache,
        )


def test_decode_rejects_no_mask_for_more_than_one_position():
    model = _two_block_model()
    tgt_in, memory, encoder_mask, _ = _teacher_forced(model=model)
    with pytest.raises(expected_exception=ValueError, match="x_mask may be None"):
        model.decode(
            tgt_in=tgt_in,
            pos_encoding=model.positional_embedding.pe[:SEQ_LEN],
            memory=memory,
            decoder_mask=None,
            memory_mask=encoder_mask,
        )
