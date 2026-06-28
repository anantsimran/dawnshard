# Convolutional Neural Networks

## Core Idea

A fully-connected layer connects every input pixel to every neuron. For a 28×28 image that's 784 inputs — and those weights don't know that pixel `(5, 5)` is next to pixel `(5, 6)`. Spatial structure is invisible.

CNNs fix this by sliding a small **filter** (kernel) across the image. The filter only ever looks at a local patch, and the same weights are reused at every position. This gives two things:

- **Local connectivity** — each output depends only on a small neighborhood
- **Weight sharing** — the filter that detects a vertical edge works anywhere in the image, not just top-left

______________________________________________________________________

## How One Filter Works

A filter of `kernel_size=3` is a `(3, 3)` grid of weights. At every spatial position `(i, j)` it looks at the 3×3 patch of pixels centered there, does a dot product (element-wise multiply then sum), and writes one number to the output:

```
filter weights (3×3):         patch at position (5, 5):
  w00 w01 w02                   p44 p45 p46
  w10 w11 w12       ·           p54 p55 p56
  w20 w21 w22                   p64 p65 p66

output[5, 5] = sum of (w_ij × p_ij)
```

Slide this across all positions → you get one output map the same size as the input (with `padding=1`).

______________________________________________________________________

## Multiple Channels: 1 → 16 → 32

**`conv1`** takes a single-channel input `(1, 28, 28)`.

It has 16 filters, each of shape `(1, 3, 3)`. They all slide across the same image independently, each learning a different pattern (one might detect horizontal edges, another vertical edges, etc.). Each produces a `(28, 28)` map. Together: `(16, 28, 28)`.

After `pool`: `(16, 14, 14)`.

**`conv2`** takes that 16-channel tensor as input.

Each of its 32 filters has shape `(16, 3, 3)` — it spans **all 16 channels at once**. At each position `(i, j)` it takes the full `16 × 3 × 3` volume (144 numbers), dot products it down to a single number, and produces one `(14, 14)` map. 32 filters → `(32, 14, 14)`.

```
input:  (16, 14, 14)

filter_0:  (16, 3, 3)  →  one (14, 14) map   ← output channel 0
filter_1:  (16, 3, 3)  →  one (14, 14) map   ← output channel 1
...
filter_31: (16, 3, 3)  →  one (14, 14) map   ← output channel 31

output: (32, 14, 14)
```

After `pool`: `(32, 7, 7)`.

Each filter learns to **combine** the 16 input features differently. One might weight "vertical edge" + "curve" channels to detect corners. Another combines different channels to detect loops. They all see the same full 16-channel input but vote independently.

______________________________________________________________________

## Flattening

After the two conv+pool steps the tensor is `(B, 32, 7, 7)`. `flatten(start_dim=1)` collapses all three non-batch dimensions:

```
32 × 7 × 7 = 1,568
```

All 32 feature maps are concatenated into one long vector. The `fc` layer then sees all 1,568 values at once.

______________________________________________________________________

## Parameter Count: `CNNClassifier`

`MaxPool2d` has no learnable parameters. Only the three layers with weights count.

**`conv1`** — `(in_channels=1, out_channels=16, kernel_size=3)`

```
weights: 16 × 1 × 3 × 3  =   144
biases:  16
─────────────────────────    160
```

**`conv2`** — `(in_channels=16, out_channels=32, kernel_size=3)`

```
weights: 32 × 16 × 3 × 3  = 4,608
biases:  32
──────────────────────────   4,640
```

**`fc`** — `(in_features=1568, out_features=10)`

```
weights: 1,568 × 10  = 15,680
biases:  10
─────────────────────  15,690
```

**Total: 160 + 4,640 + 15,690 = 20,490 parameters**

The fully-connected layer holds 77% of all parameters despite being a single matrix multiplication. Convolutions are parameter-efficient because the same `(16, 3, 3)` filter covers the whole spatial grid by sliding — but the moment you flatten and go dense, parameter count scales with spatial resolution.

______________________________________________________________________

## Visualizing the Model

Two complementary tools — one shows you shapes, one shows you gradient wiring.

### torchinfo — layer-by-layer table

`torchinfo` runs a forward pass and prints a table: one row per layer, output shape, parameter count. The fastest way to verify your architecture matches what you designed.

```python
from torchinfo import summary
summary(model=CNNClassifier(), input_size=(1, 1, 28, 28))
```

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNNClassifier                            [1, 10]                   --
├─ Conv2d: 1-1                           [1, 16, 28, 28]           160
├─ MaxPool2d: 1-2                        [1, 16, 14, 14]           --
├─ Conv2d: 1-3                           [1, 32, 7, 7]             4,640
├─ MaxPool2d: 1-4                        [1, 32, 7, 7]             --
├─ Linear: 1-5                           [1, 10]                   15,690
==========================================================================================
Total params: 20,490
```

### torchviz — computation graph

`visualize_model` (in `viz/model_graph.py`) uses `torchviz` to render the **autograd graph** — the sequence of operations PyTorch will run during backpropagation. It opens in your browser as an HTML file.

```python
from viz.model_graph import visualize_model
visualize_model(model=CNNClassifier(), input_shape=(1, 1, 28, 28))
```

The graph is read **bottom-up**: leaf nodes at the top are parameters (weights and biases, shown in blue). Each grey box is a backward operation — `AddmmBackward0` is a linear layer's backward pass, `ConvolutionBackward0` is a conv layer's. Orange boxes are intermediate tensors saved for the backward pass. Arrows show data dependencies: if A → B, then B's gradient computation needs A's value.

This is a lower-level view than `torchinfo` — use it when you want to verify gradient flow or debug why a parameter isn't getting gradients (it won't appear in the graph if it's detached from the output).
