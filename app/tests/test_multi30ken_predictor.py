import torch
from torch import nn
from training.dataclass.config import AttentionConfig, BlockConfig, EmbeddingConfig, EncoderConfig
from training.model.multi30ken_predictor import Multi30kEnPredictor
from training.transformer.modules import SinusoidalEmbedding

SEQ_LEN = 8
VOCAB = 300


def _config() -> EncoderConfig:
    return EncoderConfig(
        d_model=16,
        embedding=EmbeddingConfig(seq_length=SEQ_LEN, dropout=0.1),
        num_encoders=2,
        encoder=BlockConfig(attention=AttentionConfig(h=2)),
    )


def test_forward_returns_one_row_per_position():
    model = Multi30kEnPredictor(model_config=_config(), vocab_size=VOCAB).eval()
    ids = torch.randint(low=0, high=VOCAB, size=(3, SEQ_LEN))
    assert model(ids).shape == (3 * SEQ_LEN, VOCAB)
    assert len(model.attention_map["encoder"]) == 2  # noqa: NAR001


def test_switched_off_components_become_identity():
    config = _config().with_overrides(
        overrides={"embedding": {"position_encoding": False}, "final_norm": False}
    )
    model = Multi30kEnPredictor(model_config=config, vocab_size=VOCAB)
    assert type(model.position_embedding) is nn.Identity
    assert type(model.final_norm) is nn.Identity


def test_encoder_norm_picks_every_norm():
    config = _config().with_overrides(overrides={"encoder": {"norm": "rmsnorm"}})
    model = Multi30kEnPredictor(model_config=config, vocab_size=VOCAB)
    assert type(model.final_norm) is nn.RMSNorm
    assert type(model.encoder_blocks[0].ffn_norm) is nn.RMSNorm
    assert type(model.position_embedding) is SinusoidalEmbedding
