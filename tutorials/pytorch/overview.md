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

Every tutorial, and the section that answers a specific question, is listed in the [tutorials index](../index.md).

______________________________________________________________________

## References

- PyTorch autograd docs: pytorch.org/docs/stable/notes/autograd.html
- PyTorch data docs: pytorch.org/docs/stable/data.html
- Baydin et al., *"Automatic Differentiation in Machine Learning: a Survey"* (JMLR 2018)
- Bengio et al., *"Estimating or Propagating Gradients Through Stochastic Neurons"* (2013)
- Goodfellow, Bengio & Courville, *Deep Learning* (2016), Ch. 5–6
