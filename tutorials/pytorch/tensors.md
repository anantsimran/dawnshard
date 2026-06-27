# Tensors

## Tensors vs NumPy

| | NumPy array | PyTorch tensor |
|---|---|---|
| Device | CPU only | CPU or GPU/accelerator |
| Autograd | none | tracks gradients via `requires_grad` |
| API | `np.*` | mostly mirrors NumPy |

Tensors are NumPy ndarrays with two extra capabilities: GPU placement and gradient tracking. The API is intentionally similar — if you know NumPy you already know most of the tensor API.

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
