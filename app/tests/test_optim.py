import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW
from training.dataclass.config import OptimizerConfig, SchedulerConfig
from training.train.optim import OPTIMIZERS, build_optimizer, build_scheduler


@pytest.mark.parametrize(
    argnames=("name", "expected_type"),
    argvalues=[("adam", Adam), ("adamw", AdamW), ("sgd", SGD)],
)
def test_build_optimizer_uses_the_named_class_and_settings(name, expected_type):
    model = nn.Linear(in_features=2, out_features=1)
    optimizer = build_optimizer(
        config=OptimizerConfig(name=name, lr=0.5, weight_decay=0.1),
        params=model.parameters(),
    )
    assert type(optimizer) is expected_type  # noqa: NAR001
    assert optimizer.param_groups[0]["lr"] == 0.5
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_every_optimizer_name_has_a_class():
    assert set(OPTIMIZERS) == {"adam", "adamw", "sgd"}  # noqa: NAR001


def test_cosine_scheduler_decays_to_zero_on_the_last_epoch():
    weight = nn.Parameter(data=torch.zeros(1))  # noqa: NAR001
    optimizer = SGD(params=[weight], lr=1.0)
    scheduler = build_scheduler(
        config=SchedulerConfig(name="cosine"), optimizer=optimizer, num_epochs=4
    )
    for _ in range(4):  # noqa: NAR001
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(expected=0.0, abs=1e-12)


@pytest.mark.parametrize(argnames="name", argvalues=["cosine", "warmup_cosine", "one_cycle"])
def test_every_schedule_peaks_at_the_optimizer_lr_and_ends_below_it(name):
    weight = nn.Parameter(data=torch.zeros(1))  # noqa: NAR001
    optimizer = SGD(params=[weight], lr=1.0)
    scheduler = build_scheduler(
        config=SchedulerConfig(name=name), optimizer=optimizer, num_epochs=10
    )
    lrs = []
    for _ in range(10):  # noqa: NAR001
        lrs.append(scheduler.get_last_lr()[0])  # noqa: NAR001
        optimizer.step()
        scheduler.step()
    assert max(lrs) == pytest.approx(expected=1.0)  # noqa: NAR001
    assert lrs[-1] < 1.0
