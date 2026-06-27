# PyTorch — Overview

## What PyTorch Is

Three things:

- **Tensors** — n-dimensional arrays (like `np.ndarray`) that can live on a GPU and participate in automatic differentiation
- **Autograd** — a runtime that records every operation into a graph so it can compute derivatives on demand
- **`nn`** — a class hierarchy for organizing model parameters and composing layers

PyTorch uses a **define-by-run** approach: the computation graph is built dynamically as operations execute, not compiled upfront. Normal Python control flow (`if`, `for`) works inside a model — debugging feels like ordinary Python.

______________________________________________________________________

## The 5 Building Blocks

| What | PyTorch concept | Your job |
|---|---|---|
| Model | `nn.Module` subclass | define `__init__` + `forward()` |
| Data | `Dataset` + `DataLoader` | wrap your data |
| Loss | `nn.CrossEntropyLoss`, `nn.MSELoss`, etc. | pick one |
| Optimizer | `torch.optim.Adam`, etc. | configure lr |
| Loop | plain Python `for` loop | write it yourself |

______________________________________________________________________

## Where to Go Next

| Topic | File |
|---|---|
| Tensors vs NumPy, device placement, dtypes | [tensors.md](tensors.md) |
| Autograd, gradients, chain rule, differentiability | [grad_and_descent.md](grad_and_descent.md) |
| `nn.Module`, parameters, multi-layer networks, idioms | [nn_module_and_multi_layer_networks.md](nn_module_and_multi_layer_networks.md) |
| Training loop, epoch aggregation, structuring (functional core / Trainer) | [training_loop.md](training_loop.md) |
| Optimizer/model/scheduler decoupling, param aliasing, LR schedules | [optimizer_and_scheduler.md](optimizer_and_scheduler.md) |
| Debugging, shape checks, graph visualization | [inspecting_your_model_best_practices.md](inspecting_your_model_best_practices.md) |

______________________________________________________________________

## References

- PyTorch autograd docs: pytorch.org/docs/stable/notes/autograd.html
- PyTorch data docs: pytorch.org/docs/stable/data.html
- Baydin et al., *"Automatic Differentiation in Machine Learning: a Survey"* (JMLR 2018)
- Bengio et al., *"Estimating or Propagating Gradients Through Stochastic Neurons"* (2013)
- Goodfellow, Bengio & Courville, *Deep Learning* (2016), Ch. 5–6
