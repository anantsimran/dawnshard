# ReLU and Dead Neurons

`ReLU(x) = max(0, x)` has gradient `0` for every negative input. The natural worry: doesn't that stop backprop?

No. But the worry is pointing at a real problem, just a narrower one than "backprop stops."

______________________________________________________________________

## Why Backprop Doesn't Stop

The gate is **per-neuron** and **per-example**, and backprop through a layer is a **sum over paths**.

For hidden unit `j` with pre-activation `z_j = Σ_i w_ji·x_i + b_j` and `h_j = max(0, z_j)`:

$$\\frac{\\partial L}{\\partial x_i} = \\sum_j \\frac{\\partial L}{\\partial h_j}\\cdot \\mathbf{1}[z_j > 0]\\cdot w\_{ji}$$

The indicator kills **term `j`** of the sum, not the whole sum. As long as one unit in the layer is on, gradient reaches the layer below.

**Concrete:** a layer of 3 units, upstream gradients `∂L/∂h = [2, −1, 3]`, pre-activations `z = [−0.4, 1.2, 0.8]`, and weights into `x_1` of `w_·1 = [0.5, −0.2, 0.9]`:

$$\\frac{\\partial L}{\\partial x_1} = \\underbrace{2(0)(0.5)}\_{\\text{dead}} + (-1)(1)(-0.2) + 3(1)(0.9) = 2.9$$

Unit 1 contributed nothing. The signal still got through at full strength via the other two.

Two more reasons it self-heals:

- **Per-example.** With a batch of 64, unit 1 might be negative on 20 examples and positive on 44. The weight gradient is summed over the batch, so the unit still updates.
- **The gate moves.** Nothing pins `z_j` negative. Updates to earlier layers change `x`, so a unit that is silent this step can be active next step. At He initialization about half the units in a layer are on for any given example, and *which* half changes from example to example and step to step. Glorot, Bordes & Bengio (2011) argued this sparsity, with true zeros, is a feature.

______________________________________________________________________

## When a Unit Really Dies

If `z_j < 0` for **every input in the dataset**, the unit is permanently dead. Its own incoming weights get gradient

$$\\frac{\\partial L}{\\partial w\_{ji}} = \\frac{\\partial L}{\\partial h_j}\\cdot 0 \\cdot x_i = 0$$

on every example, forever. Nothing can revive it: the only thing that could move `z_j` is a gradient that is structurally zero. This is the **dying ReLU** problem.

**Usual cause:** one oversized update (learning rate too high) shoves the unit's weights or bias far negative. Bad initialization does it too. Lu et al. (2019) show that the probability of a whole network being *born dead* goes to 1 as depth grows, and to 0 as width grows at fixed depth.

Mitigations, roughly in order of how often you'll reach for them:

| Fix | Mechanism |
|---|---|
| He initialization | Keeps the variance of `z` sane at init, so units start about half on |
| Lower LR / gradient clipping | Prevents the single oversized update that pushes weights or bias far negative |
| BatchNorm before activation | Re-centers `z` over each batch, so it's hard for a unit to end up negative on every input |
| Leaky ReLU, `max(0.01x, x)` | Gradient is `0.01` for `x < 0`, never `0`, so there's always a way back |
| GELU / SiLU | Smooth, with a small but nonzero gradient for negative `x` |

GELU is covered in more depth in [transformer/norm.md](../transformer/norm.md#q-relu-or-gelu-inside-the-ffn).

______________________________________________________________________

## The `x = 0` Point

Genuinely non-differentiable, and genuinely irrelevant. It's a single point, and a floating-point `z` essentially never lands exactly on `0.0`. Frameworks just pick a subgradient: PyTorch and TensorFlow both return `0`. See [grad_and_descent.md](grad_and_descent.md#case-2-differentiable-almost-everywhere-kinks) for the PyTorch check.

______________________________________________________________________

## References

- Nair & Hinton, *"Rectified Linear Units Improve Restricted Boltzmann Machines"* (ICML 2010). Introduces rectified linear units, in RBMs.
- Glorot, Bordes & Bengio, *"Deep Sparse Rectifier Neural Networks"* (AISTATS 2011). The case for rectifiers in deep networks, including sparse representations with true zeros.
- Maas, Hannun & Ng, *"Rectifier Nonlinearities Improve Neural Network Acoustic Models"* (2013). Leaky ReLU.
- He et al., *"Delving Deep into Rectifiers"* (2015). He initialization and PReLU. arXiv:1502.01852
- Lu et al., *"Dying ReLU and Initialization: Theory and Numerical Examples"* (2019). Formal treatment of born-dead networks and how the probability scales with depth and width. arXiv:1903.06733
- Stanford CS231n notes, *"Neural Networks Part 1: Setting up the Architecture"*. The standard practitioner framing of dying ReLU. cs231n.github.io/neural-networks-1/
