# DataLoader — Batching and the Dataset Abstraction

**The story of one example:** the dataset retrieves it, a transform prepares it, the loader groups it with other examples, the model produces logits, and the loss reduces the batch to a scalar. Keep each job separate when revising this page.

```
dataset[i] → (image: C×H×W, label: scalar)
collate B items → images: (B,C,H,W), labels: (B,)
model → logits: (B,num_classes)
loss → one scalar (with the default mean reduction)
```

Read [the two abstractions](#two-abstractions--keep-them-separate) and [how a batch becomes a scalar loss](#how-a-batch-becomes-a-scalar-loss) for the main flow; the intervening sections explain image transforms and normalization.

______________________________________________________________________

## Why mini-batches?

- **Full dataset (batch GD):** true gradient, but one update per full pass and memory blows up
- **One sample (pure SGD):** frequent updates, but jittery convergence
- **Mini-batch (common compromise):** for roughly independent examples, gradient estimate variance shrinks approximately like `1/B`, so quadrupling batch size halves the standard deviation of the noise. Correlation and dataset size affect the real gain.

## Two abstractions — keep them separate

| | Job | Analogy |
|---|---|---|
| `Dataset` | "give me item `i`" — random access to one sample | a list/array |
| `DataLoader` | batching, shuffling, parallel loading | an iterator that groups the list |

**`Dataset`** — subclass it, implement two methods:

```python
class MyDataset(Dataset):
    def __len__(self):           # total number of samples
    def __getitem__(self, idx):  # return (x, y) for one index
```

**`DataLoader`** — configure, don't subclass:

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

It calls `__getitem__` under the hood and collates `B` samples into a batch. With simple tensor samples, default collation stacks each field and adds a leading batch dimension `(B, ...)`:

```python
for xb, yb in loader:   # xb: (B, features), yb: (B, ...)
    # 5-step loop here
```

**`shuffle=True` for training, `shuffle=False` for validation/test.** If data is ordered (e.g., all class-0 then all class-1), consecutive batches are biased. Shuffling each epoch decorrelates batches.

______________________________________________________________________

## Transforms — Format and Normalization Pipeline

`transforms` is a pipeline that processes raw data into what your model expects. Two jobs:

### Format conversion — `ToTensor()`

```python
# Raw image: PIL Image, pixel values 0–255
# After ToTensor():
#   - converts to torch.Tensor
#   - for an 8-bit PIL image, scales values from [0, 255] → [0.0, 1.0]
#   - reshapes to (C, H, W)

transforms.ToTensor()
```

This is the classic `torchvision.transforms` API. In the newer `transforms.v2` API, the same idea is split into [`ToImage()` and `ToDtype(torch.float32, scale=True)`](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html). Scaling depends on the input image type; do not assume every tensor conversion divides by 255.

Neural nets train better on small float inputs than large integers — the scale directly affects gradient magnitude (see [grad_and_descent.md](grad_and_descent.md#why-centered-inputs-stabilize-gradients)).

### Normalization — `Normalize(mean, std)`

```python
# Shifts each pixel so distribution is centered at 0
# (pixel - mean) / std   — applied independently per channel

transforms.Normalize((mean_ch1,), (std_ch1,))
```

**Why a tuple?** One value per channel. A single-channel image takes `(mean,)`. RGB takes three:

```python
transforms.Normalize((0.485, 0.456, 0.406),   # ImageNet R, G, B means
                     (0.229, 0.224, 0.225))    # R, G, B stds
```

### Compose chains them

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
# Applied in order, automatically, to every sample the DataLoader yields
```

### Why it's the standard pattern — lazy + train/test split

Transforms are applied **lazily** — per sample as batches load, not upfront. This makes it natural to use different transforms for training vs evaluation:

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),   # augmentation — training only
    transforms.ToTensor(),
    transforms.Normalize(...)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),           # no augmentation for test
    transforms.Normalize(...)
])
```

______________________________________________________________________

## How Augmentations Work

`Dataset.__getitem__(i)` is the hook point. The transform is stored on the dataset and called inside `__getitem__`:

```python
# Simplified torchvision internals
class SomeDataset(Dataset):
    def __getitem__(self, i):
        img, label = self.raw_data[i]      # raw, untouched on disk
        if self.transform:
            img = self.transform(img)      # called fresh, every access
        return img, label
```

Three consequences:

- **Stateless & per-call** — the raw bytes never change. Each `__getitem__` call produces a new transformed copy. A `RandomRotation` rolls a new angle *every epoch* → the model sees a slightly different image each time → free data variety without changing the dataset.
- **Lazy** — nothing runs until the DataLoader pulls index `i`. Memory holds raw data only; transformed tensors are ephemeral, created and discarded per batch.
- **Composable** — `Compose` is just function composition. Each transform is a callable:

```python
class Compose:
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
```

A list of callables applied in sequence — that's it.

______________________________________________________________________

## PIL Image

PIL = **Python Imaging Library** (the `Pillow` fork). It's a common Python object for representing an image in memory before tensor conversion. Many torchvision image datasets return PIL Images before a transform converts them.

```python
from torchvision import datasets

raw = datasets.MNIST(root="data", train=True, download=True)  # no transform
img, label = raw[0]

print(type(img))    # <class 'PIL.Image.Image'>
print(img.size)     # (28, 28)  — note: PIL is (W, H), not (H, W)
print(img.mode)     # 'L'  → 'L' means 8-bit grayscale
```

`type(img)` is how you confirm. After `ToTensor()` it becomes `torch.Tensor`.

______________________________________________________________________

## Computing Normalization Stats

**Can you compute mean/std at runtime?** Yes, from the training set. Weight every pixel equally; averaging batch standard deviations is not the standard deviation of the whole dataset, and averaging batch means equally is wrong when the final batch is smaller.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ds = datasets.MNIST(root="data", train=True, download=True,
                    transform=transforms.ToTensor())
loader = DataLoader(ds, batch_size=1000)

total = torch.zeros(1)   # one entry per image channel (MNIST has one)
total_sq = torch.zeros(1)
count = 0
for xb, _ in loader:     # xb: (B,C,H,W)
    total += xb.sum(dim=(0, 2, 3))
    total_sq += xb.square().sum(dim=(0, 2, 3))
    count += xb.shape[0] * xb.shape[2] * xb.shape[3]
mean = total / count
std = (total_sq / count - mean.square()).clamp_min(0).sqrt()
```

**Do train and test stats differ?** Slightly — they're different image sets. **But always use training stats for both.** At inference you may have only one image; you can't compute meaningful stats from it. Test data is treated as unseen, so you normalize it with numbers derived only from training. Using test stats on test data is data leakage.

This is why constants like MNIST's 0.1307/0.3081 are hardcoded — computed once on train, reused everywhere.

______________________________________________________________________

## What are Logits?

Raw, unnormalized scores output by the model — one number per class, before any probability conversion.

**Example** — 3 classes (cat, dog, bird):

```
logits = [2.1, 0.3, 1.5]
           cat  dog  bird
```

These aren't probabilities — they don't sum to 1 and can be negative. To get probabilities, you pass them through softmax:

```
softmax([2.1, 0.3, 1.5]) ≈ [0.58, 0.10, 0.32]  ← now sums to 1
```

**Why keep them as logits and not convert first?**

`CrossEntropyLoss` takes raw logits and does the softmax internally — this is numerically more stable than doing `softmax → log → loss` yourself. So the convention is: **model outputs logits, loss function handles the conversion.**

______________________________________________________________________

## How a Batch Becomes a Scalar Loss

`loss = criterion(logits, yb)  # logits: (B, C), yb: (B,) → scalar`

Say `B=3`, `C=4` (3 samples, 4 classes). `CrossEntropyLoss` does this in two steps:

**Step 1 — loss per sample** (softmax + log + pick the correct class):

```
loss_0 = 0.43
loss_1 = 0.12
loss_2 = 0.67
```

**Step 2 — reduce to scalar:**

```
loss = mean(0.43, 0.12, 0.67) = 0.407   ← this is what .backward() sees
```

**Averaged, not summed — and that's the default.** PyTorch loss functions have a `reduction` argument:

```python
nn.CrossEntropyLoss(reduction='mean')   # default — divides by B
nn.CrossEntropyLoss(reduction='sum')    # sums instead
nn.CrossEntropyLoss(reduction='none')   # returns (B,), no reduction
```

**Why mean and not sum?** With sum, your loss (and gradients) scale with batch size — doubling `B` doubles the gradient magnitude. With mean, gradient scale stays constant regardless of batch size, so your learning rate stays stable.

## Common confusions

- `Dataset` supplies individual samples; `DataLoader` decides order, batching, and worker loading. A map-style dataset does not need to return a prebuilt batch.
- A random training transform is drawn on access; evaluation should use a stable transform. Both splits use normalization statistics calculated from **training data only**.
- Logits are scores, not probabilities. `CrossEntropyLoss` expects logits and class indices, and applies the needed log-softmax internally.

## Check your understanding

1. If images are `(3, 28, 28)` and the loader collates 32 of them, what shape reaches the model?
1. Why is averaging batch standard deviations insufficient for a dataset-wide standard deviation?
1. Why should test images use the training set's mean and standard deviation?

<details markdown="1"><summary>Answers</summary>

1. `(32, 3, 28, 28)`; the loader adds a leading batch axis.
1. Standard deviation depends on distances from the **global** mean; within-batch deviations omit differences between batch means.
1. Evaluation must treat test data as unseen, and an individual inference example may not support a meaningful new estimate.

</details>

**One-minute recap:** the dataset retrieves one item, transforms prepare it, the loader collates batches, and the loss reduces per-example errors. Estimate normalization statistics from training pixels, then reuse them.
