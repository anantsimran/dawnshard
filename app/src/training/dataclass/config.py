"""config.py -- the configs a run is built from.

`TrainConfig` is one run, the root of a YAML file: the architecture (`model`, a
`ModelConfig`), the epochs and seeds `fit` reads (`fit`, a `FitConfig`), the optimizer,
scheduler, and dataloader settings the caller builds `TrainState` and the loaders
from, and the dataset settings (`data`). A model narrows `model` and `data` to its own
types in a subclass. Every config is a `Config`, so it round-trips through a plain
dict and loads from YAML.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Self, get_args, get_type_hints

import yaml


@dataclass(frozen=True)
class Config:
    """Base for configs that are plain data and round-trip through a dict.

    A subclass is a frozen dataclass whose fields are str, int, float, bool, or another
    `Config`. No tuples, enums, or classes, so `to_dict` gives what JSON and
    `yaml.safe_load` give back, and `from_dict` reads either. Lists of those are fine.
    Names, such as a norm or an optimizer, are strings; the code that reads the config
    maps them to classes.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return the config as nested plain dicts."""
        return dataclasses.asdict(obj=self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build the config from `to_dict`'s output, nested configs included.

        A float field also takes an int or a numeric string, because YAML 1.1 reads
        `1e-3` as the string "1e-3". No other value is coerced or checked; a wrong
        type surfaces where the config is used.

        Raises:
            ValueError: If `data` has a key that is not a field of the config.
            TypeError: If a field without a default is missing, from the constructor.
        """
        fields = {field.name for field in dataclasses.fields(class_or_instance=cls)}
        unknown = data.keys() - fields
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")  # noqa: NAR001
        hints = get_type_hints(obj=cls)
        kwargs = {name: _read_field(hint=hints[name], value=value) for name, value in data.items()}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Build the config from a YAML file, as `from_dict` reads its contents."""
        with open(file=path) as config_file:
            return cls.from_dict(data=yaml.safe_load(stream=config_file))

    def with_overrides(self, overrides: Mapping[str, Any]) -> Self:
        """Return a copy with `overrides` merged over `to_dict()`, nested dicts included.

        A nested dict merges into the matching nested config, so
        `{"model": {"d_model": 256}}` changes that one field and keeps the rest. Any
        other value, None and lists included, replaces the field outright.

        Raises:
            ValueError: If `overrides` names a key the config does not have.
        """
        return type(self).from_dict(data=_merge(base=self.to_dict(), overrides=overrides))  # noqa: NAR001


def _read_field(hint: Any, value: Any) -> Any:
    """Return `value` as the field typed `hint` holds it.

    A dict becomes the nested `Config` the hint names, `Optional` included, and a float
    field's int or string becomes a float. Anything else is returned as is.
    """
    if value is None:
        return None
    candidates = get_args(hint) or (hint,)  # noqa: NAR001
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, Config):  # noqa: NAR001
            return candidate.from_dict(data=value)
    if float in candidates and isinstance(value, (int, str)) and not isinstance(value, bool):  # noqa: NAR001
        return float(value)  # noqa: NAR001
    return value


def _merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return `base` with `overrides` merged in, recursing where both hold a dict."""
    merged = dict(base)  # noqa: NAR001
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):  # noqa: NAR001
            merged[key] = _merge(base=merged[key], overrides=value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ModelConfig(Config):
    """Base for architecture configs; the model that reads one maps its names."""


type NormName = Literal["layernorm", "rmsnorm"]


@dataclass(frozen=True)
class AttentionConfig(ModelConfig):
    # Heads per attention sublayer; d_model must divide by it.
    h: int
    # RMSNorm queries and keys per head before the dot product.
    qk_norm: bool = False


@dataclass(frozen=True)
class BlockConfig(ModelConfig):
    attention: AttentionConfig
    # FFN hidden width as a multiple of d_model.
    ffn_multiplier: int = 4
    # On each sublayer's output, before its residual add.
    dropout: float = 0.1
    norm: NormName = "layernorm"


@dataclass(frozen=True)
class EmbeddingConfig(ModelConfig):
    # Longest L the model accepts on this side.
    seq_length: int
    # On the token embeddings. Where it sits relative to the position encoding is the
    # model's choice; see its docstring.
    dropout: float
    # Add the fixed sin/cos table. False leaves attention as the only thing that mixes
    # positions, so the model cannot tell word order.
    position_encoding: bool = True


@dataclass(frozen=True)
class TransformerConfig(ModelConfig):
    # The one residual width: cross-attention reads encoder output in the decoder's
    # width, so the two stacks cannot differ in it. Vocab sizes are not here; they
    # follow from the tokenizers.
    d_model: int
    src: EmbeddingConfig
    tgt: EmbeddingConfig
    num_encoders: int
    num_decoders: int
    # Every block in a stack is built from its side's setting.
    encoder: BlockConfig
    decoder: BlockConfig


@dataclass(frozen=True)
class EncoderConfig(ModelConfig):
    # An embedding, a stack of pre-norm encoder blocks, and a head the model picks.
    d_model: int
    embedding: EmbeddingConfig
    num_encoders: int
    # Every block in the stack is built from it, and the final norm is its norm class.
    encoder: BlockConfig
    # False sends the last block's unnormed residual straight to the head.
    final_norm: bool = True


type OptimizerName = Literal["adam", "adamw", "sgd"]
type SchedulerName = Literal["cosine", "warmup_cosine", "one_cycle"]


@dataclass(frozen=True)
class OptimizerConfig(Config):
    # `train.optim.OPTIMIZERS` maps the name to a torch optimizer class.
    name: OptimizerName
    lr: float
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig(Config):
    # `fit` steps the scheduler once per epoch, so its horizon is `num_epochs`.
    # `train.optim.build_scheduler` maps the name to a schedule.
    name: SchedulerName


@dataclass(frozen=True)
class DataloaderConfig(Config):
    batch_size: int


@dataclass(frozen=True)
class DataloadConfig(Config):
    """Base for a dataset's settings; the code that builds the loaders names the fields."""


@dataclass(frozen=True)
class EarlyStoppingConfig(Config):
    # Counted in validation passes, not epochs: `fit` stops once `patience` passes
    # in a row fail to beat the best `val_loss` by more than `min_delta`.
    patience: int
    min_delta: float = 0.0


@dataclass(frozen=True)
class FitConfig(Config):
    # `fit` and `profiled_fit` read every field, and the history file records them.
    num_epochs: int
    val_seed: int
    test_seed: int
    # A YAML leaves it None; the caller draws a fresh one per run and fills it in
    # with `dataclasses.replace` before building the model. `fit` raises if it is
    # still None.
    train_seed: Optional[int] = None
    # None validates after every epoch. Never empty: `fit` checkpoints after each
    # validation pass, so a run with none would leave no weights behind.
    val_epoch_list: Optional[list[int]] = None
    # 1-based number of the first epoch to run.
    epoch_start: int = 1
    # Names the run's checkpoint folder, `CHECKPOINTS_PATH / run_uuid`.
    run_uuid: Optional[str] = None
    early_stopping: Optional[EarlyStoppingConfig] = None

    def __post_init__(self) -> None:
        """Reject an empty `val_epoch_list`.

        Raises:
            ValueError: If `val_epoch_list` is an empty list.
        """
        if self.val_epoch_list is not None and not self.val_epoch_list:
            raise ValueError(  # noqa: NAR001
                "val_epoch_list is empty; leave it None to validate every epoch"
            )


@dataclass(frozen=True)
class TrainConfig(Config):
    """One run, as its YAML file holds it.

    `fit` reads only `fit`. The caller builds the model from `model`, the optimizer and
    scheduler from `optimizer` and `scheduler`, and the loaders from `dataloader` and
    `data`. A model's subclass narrows `model` and `data` to its own types, which is
    what lets `from_dict` build them.
    """

    model: ModelConfig
    fit: FitConfig
    optimizer: OptimizerConfig
    dataloader: DataloaderConfig
    data: DataloadConfig
    # None trains at a constant learning rate.
    scheduler: Optional[SchedulerConfig] = None
