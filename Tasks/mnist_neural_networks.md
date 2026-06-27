# MNIST Neural Networks — Project Plan

## Problem Statement

Classify 28×28 grayscale handwritten digit images into 10 classes (0–9) using three progressively more sophisticated architectures on the same dataset — so the MLP-vs-CNN comparison is clean and architecture is the only variable.

______________________________________________________________________

## Success Criteria

| Deliverable | Criterion | How to verify |
|---|---|---|
| **Sequential MLP** | ~97% test accuracy | Evaluate on MNIST test set after a few epochs |
| **Explicit `forward()` MLP** | Similar or slightly better than D1 | Same test set evaluation |
| **CNN** | ~99% test accuracy, fewer params than MLP | Same test set; print `sum(p.numel() for p in model.parameters())` |
| **Visualization** | Loss, CPU, GPU usage visible during training | Charts render without errors |
| **CPU vs GPU** | Documented conditions for each | Written notes + code demonstrating device selection |

______________________________________________________________________

## Dataset

`torchvision.datasets.MNIST` — download via:

```python
from torchvision import datasets, transforms
datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
```

`ToTensor()` produces tensors of shape `(1, 28, 28)` — the leading `1` is the channel dimension (grayscale).

______________________________________________________________________

## Deliverable 1 — `nn.Sequential` MLP

**Architecture:**

```
(B, 1, 28, 28) → Flatten → (B, 784) → Linear(784, 128) → ReLU → Linear(128, 10)
```

- Loss: `nn.CrossEntropyLoss`
- Optimizer: `Adam`
- Target: ~97% test accuracy

______________________________________________________________________

## Deliverable 2 — Explicit `forward()` MLP

Same task, rewritten in Idiom B (explicit `forward()`). Adds a second hidden layer:

```
Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 10)
```

Goal: muscle memory for the `def forward` pattern required by Transformers.

______________________________________________________________________

## Deliverable 3 — Minimal CNN

**Architecture (explicit `forward()`):**

```
Conv2d(1, 16, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)   # (B, 16, 14, 14)
Conv2d(16, 32, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)  # (B, 32, 7, 7)
Flatten                                                          # (B, 1568)
Linear(1568, 10)
```

Shape derivation: `28 →pool→ 14 →pool→ 7`, so `32 × 7 × 7 = 1568`.

- Target: ~99% test accuracy
- Key concepts to understand: weight sharing, spatial locality, translation invariance, why pooling helps

______________________________________________________________________

## Deliverable 4 — Visualization

Set up tooling to observe training in real time:

- **Loss curve** — per-epoch training and validation loss
- **CPU usage** — utilization % during training
- **GPU usage** — utilization % and memory during training (if available)

Options: `matplotlib` for loss plots; `psutil` for CPU; `nvidia-smi` / `torch.cuda` for GPU; or use `tensorboard` / `wandb` for all-in-one.

______________________________________________________________________

## Deliverable 5 — CPU vs GPU Conditions

Document and demonstrate:

- How PyTorch selects the device (`torch.cuda.is_available()`, `.to(device)`)
- When training runs on CPU (no CUDA, small model/data, MPS on Apple Silicon)
- When training runs on GPU (CUDA available, explicit `.to("cuda")`)
- Performance comparison on MNIST for each device

______________________________________________________________________

## Prerequisites — PyTorch Basics

Complete these before starting the MNIST deliverables. Each step is small enough to finish in one sitting.

1. **Toy dataset** — linear regression: `y = 2x + noise`. Just tensors, no library needed yet.
1. **2-layer `nn.Module`** — one `nn.Linear`, one activation. Implement `forward()`.
1. **`nn.MSELoss` + `torch.optim.SGD`** — simplest loss + optimizer combo.
1. **5-step loop from memory** — don't look it up. Get it wrong, debug it.
1. **Plot loss vs. epoch** — should decrease. If it explodes or flatlines, you've hit a real training issue.
1. **Wrap data in `Dataset`** — implement `__len__`, `__getitem__`.
1. **Feed to `DataLoader`** — `batch_size=16`, `shuffle=True`. Print `xb.shape` inside the loop once.
1. **2-layer MLP** — `Linear → ReLU → Linear`, both idioms (Sequential and explicit `forward()`).
1. **Nested epoch/batch loop** with `nn.CrossEntropyLoss` on a 2-class problem.

______________________________________________________________________

## Todo List

### Setup

- [ ] Install dependencies: `torch`, `torchvision`, `matplotlib`, `psutil`
- [ ] Download MNIST dataset and verify DataLoader output shapes

### Models

- [ ] Implement Deliverable 1: `nn.Sequential` MLP, train, evaluate (~97%)
- [ ] Implement Deliverable 2: Explicit `forward()` deep MLP, train, evaluate
- [ ] Implement Deliverable 3: CNN, train, evaluate (~99%)
- [ ] Add `model.eval()` + `torch.no_grad()` to all test accuracy passes

### Visualization -maybe use wandb?

- [ ] Plot training loss per epoch for all three models
- [ ] Log CPU usage during training
- [ ] Log GPU usage and memory during training (if CUDA/MPS available)
- [ ] Render all charts in a single comparison view

### CPU vs GPU

- [ ] Write device-selection utility (`get_device()`)
- [ ] Run training on CPU and GPU (if available) and compare wall-clock time
- [ ] Document conditions: when each device is chosen, why, and the performance delta
