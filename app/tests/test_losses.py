import torch
import torch.nn.functional as F
from training.train.losses import ChunkedCrossEntropy


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(50, 8, dtype=torch.float64))

    @property
    def lm_head_weight(self):
        return self.weight


def test_chunked_equals_full_cross_entropy_in_loss_and_gradients():
    torch.manual_seed(seed=0)
    head = Head()
    hidden = torch.randn(37, 8, dtype=torch.float64, requires_grad=True)
    target = torch.randint(low=0, high=50, size=(37,))
    target[5] = -100
    full = F.cross_entropy(hidden @ head.weight.T, target)
    full.backward()
    expected = (hidden.grad.clone(), head.weight.grad.clone())
    hidden.grad, head.weight.grad = None, None
    chunked = ChunkedCrossEntropy(head=head, chunk_size=8)(hidden, target)
    chunked.backward()
    assert abs(chunked.item() - full.item()) < 1e-6
    torch.testing.assert_close(actual=hidden.grad, expected=expected[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(actual=head.weight.grad, expected=expected[1], atol=1e-6, rtol=0)


def test_head_is_not_registered_as_a_submodule():
    head = Head()
    loss = ChunkedCrossEntropy(head=head)
    assert list(loss.parameters()) == []
