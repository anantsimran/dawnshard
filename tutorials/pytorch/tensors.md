# Tensors

**Question to keep in mind:** what must you know about a tensor before combining it with another? Its **shape**, **dtype**, **device**, and whether autograd is tracking it.

Imagine a batch of 4 RGB images, each 28×28 pixels: its shape is `(4, 3, 28, 28)`. Indexing one image removes the batch slot and leaves `(3, 28, 28)`. Moving it to a GPU changes its device, not its shape. Setting `requires_grad` changes how later operations are tracked, not its values.

______________________________________________________________________

## Tensors vs NumPy

| | NumPy array | PyTorch tensor |
|---|---|---|
| Device | CPU only | CPU or GPU/accelerator |
| Autograd | none | tracks gradients via `requires_grad` |
| API | `np.*` | mostly mirrors NumPy |

Tensors are conceptually similar to NumPy ndarrays, with accelerator placement and optional gradient tracking. They are distinct objects, but the APIs overlap enough that NumPy experience transfers.

______________________________________________________________________

## Creating Tensors

```python
import torch

# from data
x = torch.tensor([1.0, 2.0, 3.0])

# from NumPy — shares memory on CPU, watch for aliasing bugs
import numpy as np
arr = np.array([1.0, 2.0])
t = torch.from_numpy(arr)
arr[0] = 99.0
print(t[0])   # tensor(99.) — same memory

# built-in constructors
torch.zeros(3, 4)
torch.randn(3, 4)    # standard normal
torch.arange(10)
```

______________________________________________________________________

## Device Placement

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(3, 4).to(device)
# or create directly on device:
x = torch.randn(3, 4, device=device)
```

**Pin dtype and device at boundaries.** Mixing `float32`/`float64` or CPU/GPU either raises or silently up/down-casts. Convert once when data enters your system:

```python
x = torch.as_tensor(arr, dtype=torch.float32, device=device)
```

______________________________________________________________________

## `requires_grad`

`requires_grad=True` tells autograd to track operations on this tensor and compute gradients during `.backward()`. You rarely set this manually — model parameters (`nn.Parameter`) have it set automatically.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
x.grad   # tensor([4.])
```

See [grad_and_descent.md](grad_and_descent.md) for how the graph works.

## Common confusions

- A 1-D shape `(3,)` and a row shape `(1, 3)` hold three numbers but have different ranks; broadcasting may treat them differently. See [shapes and broadcasting](broadcasting.md).
- `.to(device)` returns a tensor on the requested device. Assign the result if later code should use it.
- `requires_grad=True` does not itself compute a gradient. A connected scalar result must be differentiated with `.backward()`.

## Check your understanding

1. What is the shape of one image selected from `(4, 3, 28, 28)` with `images[0]`?
1. Why can `torch.from_numpy(arr)` reflect a later change to `arr`?
1. Does calling `.to("cuda")` change a tensor's number of dimensions?

<details markdown="1"><summary>Answers</summary>

1. `(3, 28, 28)`: indexing removed the leading batch dimension.
1. On the CPU, they can share the same underlying memory.
1. No. Device placement changes where values live, not the shape.

</details>

**One-minute recap:** shape describes indexing, dtype describes representation, device describes location, and autograd tracking describes whether operations can contribute to gradients.
