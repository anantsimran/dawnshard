import torch
from dist_helpers import run_distributed
from torch import nn
from training.parallel.ddp import broadcast_parameters
from training.parallel.zero import Zero2AdamW, assign_owners


def test_each_parameter_has_exactly_one_owner_and_load_is_balanced():
    numels = [100, 60, 50, 40, 10, 5]
    owners = assign_owners(numels=numels, world_size=2)
    assert len(owners) == len(numels) and set(owners) <= {0, 1}
    loads = [sum(n for n, o in zip(numels, owners) if o == r) for r in (0, 1)]
    assert abs(loads[0] - loads[1]) <= max(numels)


def _model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


def _batches():
    generator = torch.Generator().manual_seed(7)
    return [(torch.randn(8, 4, generator=generator), torch.randn(8, 3, generator=generator)) for _ in range(3)]


def _check_zero2(rank, world):
    model = _model(seed=rank)
    broadcast_parameters(module=model)
    optimizer = Zero2AdamW(params=list(model.parameters()), lr=1e-2)
    reference = _model(seed=0)
    reference_optimizer = torch.optim.AdamW(params=reference.parameters(), lr=1e-2, weight_decay=0.01)
    for x, y in _batches():
        optimizer.zero_grad()
        nn.functional.mse_loss(model(x[rank * 4 : (rank + 1) * 4]), y[rank * 4 : (rank + 1) * 4]).backward()
        owned = [i for i, owner in enumerate(optimizer.owners) if owner == rank]
        held = [i for i, p in enumerate(optimizer.flat) if p.grad is not None]
        assert held == owned
        optimizer.step()
        reference_optimizer.zero_grad()
        nn.functional.mse_loss(reference(x), y).backward()
        reference_optimizer.step()
    for param, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual=param, expected=expected, atol=1e-6, rtol=1e-5)


def test_zero2_matches_adamw_and_keeps_only_owned_gradients():
    run_distributed(fn=_check_zero2)
