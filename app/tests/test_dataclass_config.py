import json

import pytest
from training.constants import CONFIGS_PATH
from training.dataclass.config import (
    AttentionConfig,
    BlockConfig,
    EarlyStoppingConfig,
    EmbeddingConfig,
    FitConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
    TransformerConfig,
)
from training.run.ag_news_classifier import AGNewsClassifierTrainConfig
from training.run.multi30k_translator import Multi30kTranslatorTrainConfig
from training.run.multi30ken_predictor import Multi30kEnPredictorTrainConfig


def _config() -> TransformerConfig:
    return TransformerConfig(
        d_model=16,
        src=EmbeddingConfig(seq_length=8, dropout=0.1),
        tgt=EmbeddingConfig(seq_length=6, dropout=0.2),
        num_encoders=2,
        num_decoders=1,
        encoder=BlockConfig(attention=AttentionConfig(h=2, qk_norm=True), norm="rmsnorm"),
        decoder=BlockConfig(attention=AttentionConfig(h=4), ffn_multiplier=2, dropout=0.0),
    )


def test_config_survives_json():
    config = _config()
    data = json.loads(s=json.dumps(obj=config.to_dict()))
    assert TransformerConfig.from_dict(data=data) == config


def test_to_dict_is_plain_nested_dicts():
    data = _config().to_dict()
    assert data["encoder"]["attention"] == {"h": 2, "qk_norm": True}


def test_from_dict_fills_defaults():
    block = BlockConfig.from_dict(data={"attention": {"h": 2}})
    assert block == BlockConfig(attention=AttentionConfig(h=2))


def test_from_dict_rejects_unknown_key():
    data = _config().to_dict()
    data["decoder"]["heads"] = 4
    with pytest.raises(expected_exception=ValueError, match="heads"):
        TransformerConfig.from_dict(data=data)


def test_from_dict_rejects_missing_nested_key():
    data = _config().to_dict()
    del data["encoder"]["attention"]["h"]
    with pytest.raises(expected_exception=TypeError):
        TransformerConfig.from_dict(data=data)


def test_with_overrides_merges_nested_dicts_and_keeps_the_rest():
    config = _config().with_overrides(overrides={"encoder": {"attention": {"h": 8}}})
    assert config.encoder.attention == AttentionConfig(h=8, qk_norm=True)
    assert config.encoder.norm == "rmsnorm"
    assert config.decoder == _config().decoder


def test_with_overrides_rejects_unknown_key():
    with pytest.raises(expected_exception=ValueError, match="heads"):
        _config().with_overrides(overrides={"encoder": {"heads": 8}})


def test_with_overrides_none_clears_an_optional_section():
    config = FitConfig(
        num_epochs=1,
        val_seed=1,
        test_seed=2,
        early_stopping=EarlyStoppingConfig(patience=2),
    )
    assert config.with_overrides(overrides={"early_stopping": None}).early_stopping is None


def test_from_dict_builds_optional_nested_configs():
    config = FitConfig.from_dict(
        data={"num_epochs": 1, "val_seed": 1, "test_seed": 2, "early_stopping": {"patience": 3}}
    )
    assert config.early_stopping == EarlyStoppingConfig(patience=3)


def test_from_yaml_reads_exponent_floats(tmp_path):
    # YAML 1.1 reads 1e-3 as the string "1e-3".
    path = tmp_path / "optimizer.yaml"
    path.write_text(data="name: adam\nlr: 1e-3\nweight_decay: 0\n")
    config = OptimizerConfig.from_yaml(path=path)
    assert config == OptimizerConfig(name="adam", lr=0.001, weight_decay=0.0)
    assert type(config.lr) is float


# Every YAML under CONFIGS_PATH and the TrainConfig it must load as. A new YAML fails
# test_every_yaml_has_a_type until it gets a row here.
YAML_TYPES: dict[str, type[TrainConfig]] = {
    "multi30k_translator.yaml": Multi30kTranslatorTrainConfig,
    "multi30ken_predictor.yaml": Multi30kEnPredictorTrainConfig,
    "ag_news_classifier.yaml": AGNewsClassifierTrainConfig,
    "ablations/multi30k_translator.yaml": Multi30kTranslatorTrainConfig,
    "ablations/multi30ken_predictor.yaml": Multi30kEnPredictorTrainConfig,
}


def test_every_yaml_has_a_type():
    paths = {path.relative_to(CONFIGS_PATH).as_posix() for path in CONFIGS_PATH.rglob("*.yaml")}  # noqa: NAR001
    assert paths == YAML_TYPES.keys()


@pytest.mark.parametrize(argnames="name", argvalues=sorted(YAML_TYPES))  # noqa: NAR001
def test_yaml_loads_and_round_trips(name):
    config = YAML_TYPES[name].from_yaml(path=CONFIGS_PATH / name)
    assert type(config.model) is not ModelConfig  # noqa: NAR001
    assert type(config).from_dict(data=config.to_dict()) == config  # noqa: NAR001
    # A YAML leaves the per-run values to the caller.
    assert config.fit.train_seed is None
    assert config.fit.run_uuid is None


def test_fit_config_rejects_an_empty_val_epoch_list():
    with pytest.raises(expected_exception=ValueError, match="val_epoch_list"):
        FitConfig(num_epochs=1, val_seed=1, test_seed=2, val_epoch_list=[])
