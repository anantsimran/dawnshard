# PyTorch — Overview

**The picture to remember:** data enters as tensors; a module turns those tensors into predictions; a loss compares predictions with targets; autograd finds how each parameter affected the loss; an optimizer changes the parameters. The next batch repeats the cycle.

| If you are revising… | Start with |
|---|---|
| What a shape, device, or gradient flag means | [Tensors](tensors.md) and [shapes and broadcasting](broadcasting.md) |
| Why a parameter receives a gradient | [Gradients and gradient descent](grad_and_descent.md) and [linear layer backward](linear_layer_backward.md) |
| How layers become a model | [`nn.Module` and multi-layer networks](nn_module_and_multi_layer_networks.md) |
| Where batches and transforms happen | [DataLoader](dataloading.md) and [random streams and workers](randomness_and_workers.md) |

This is a revision map. The examples in these pages illustrate ideas; the working code lives in the project.

______________________________________________________________________

## What PyTorch Is

Three things:

- **Tensors** — n-dimensional arrays (like `np.ndarray`) that can live on a GPU and participate in automatic differentiation
- **Autograd** — records operations involving tracked tensors into a graph so it can compute derivatives on demand
- **`nn`** — a class hierarchy for organizing model parameters and composing layers

In eager mode, PyTorch uses a **define-by-run** approach: the computation graph is built dynamically as operations execute. Normal Python control flow (`if`, `for`) works inside a model — debugging feels like ordinary Python. Compilation is an optional optimization.

For one batch, read the flow as `inputs → model → logits → loss → backward → optimizer step`. The graph belongs to that forward pass; parameters persist for the next one.

______________________________________________________________________

## The 5 Building Blocks

| What | PyTorch concept | Role in the cycle |
|---|---|---|
| Model | `nn.Module` subclass | owns parameters and maps inputs to predictions |
| Data | `Dataset` + `DataLoader` | supplies examples in batches |
| Loss | `nn.CrossEntropyLoss`, `nn.MSELoss`, etc. | turns prediction errors into a scalar |
| Optimizer | `torch.optim.Adam`, etc. | updates parameters from their gradients |
| Loop | plain Python `for` loop | repeats the cycle over batches |

______________________________________________________________________

## Where to Go Next

Every tutorial, and the section that answers a specific question, is listed in the [tutorials index](../index.md).

## Check your understanding

1. Which object persists between batches: the forward computation graph or the model's parameters?
1. Where does the optimizer get the information used to change parameters?
1. What does the loss compare?

<details markdown="1"><summary>Answers</summary>

1. Parameters persist. A new forward pass builds a new graph.
1. `backward()` writes gradients to the parameters; the optimizer reads those gradients.
1. The model's predictions and the target values.

</details>

**One-minute recap:** tensors carry values and shapes; modules own persistent state; autograd traces the current calculation; the loss and optimizer turn that trace into a parameter update.

______________________________________________________________________

## References

- PyTorch autograd docs: pytorch.org/docs/stable/notes/autograd.html
- PyTorch data docs: pytorch.org/docs/stable/data.html
- Baydin et al., *"Automatic Differentiation in Machine Learning: a Survey"* (JMLR 2018)
- Bengio et al., *"Estimating or Propagating Gradients Through Stochastic Neurons"* (2013)
- Goodfellow, Bengio & Courville, *Deep Learning* (2016), Ch. 5–6
