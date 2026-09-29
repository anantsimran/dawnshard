# ReLU and Dead Neurons

`ReLU(x) = max(0, x)` has gradient `0` for every negative input. The natural worry: doesn't that stop backprop?

For one inactive unit on one example, only that path is gated. Other active units and other examples can still contribute. The lasting problem is a unit that stays inactive over the data it sees.

**Trace to remember:** with three units, `z=[−0.4, 1.2, 0.8]`, only the first path is gated. The other two terms still add to the input gradient below.

______________________________________________________________________

## Why Backprop Doesn't Stop

The gate is **per-neuron** and **per-example**, and backprop through a layer is a **sum over paths**.

For hidden unit `j` with pre-activation `z_j = Σ_i w_ji·x_i + b_j` and `h_j = max(0, z_j)`:

∂L/∂xᵢ = Σⱼ (∂L/∂hⱼ) · 1[zⱼ > 0] · wⱼᵢ

The indicator kills **term `j`** of the sum, not the whole sum. An active unit can still pass gradient to the layer below when its upstream gradient and weight provide a signal.

**Concrete:** a layer of 3 units, upstream gradients `∂L/∂h = [2, −1, 3]`, pre-activations `z = [−0.4, 1.2, 0.8]`, and weights into `x_1` of `w_·1 = [0.5, −0.2, 0.9]`:

∂L/∂x₁ = 2(0)(0.5) + (−1)(1)(−0.2) + 3(1)(0.9) = 2.9

Unit 1 contributed nothing. The other two still contributed to the input gradient.

Two more reasons it self-heals:

- **Per-example.** With a batch of 64, unit 1 might be negative on 20 examples and positive on 44. The weight gradient is summed over the batch, so the unit still updates.
- **The gate moves.** Nothing pins `z_j` negative. Updates to earlier layers change `x`, so a unit that is silent this step can be active next step. With roughly symmetric pre-activations, about half the units may be on for a given example; the exact fraction depends on the data and initialization. Glorot, Bordes & Bengio (2011) argued this sparsity, with true zeros, is a feature.

______________________________________________________________________

## When a Unit Really Dies

If `z_j < 0` for **every input the unit sees**, its own incoming weights receive zero task gradient:

∂L/∂wⱼᵢ = (∂L/∂hⱼ) · 0 · xᵢ = 0

on every such example. With a fixed input representation, those weights cannot move through this loss, which is the **dying ReLU** problem. In a deeper network, updates to earlier shared layers can change the unit's inputs and sometimes reactivate it, so "dead" is a practical description rather than an absolute guarantee.

**Usual cause:** one oversized update (learning rate too high) shoves the unit's weights or bias far negative. Bad initialization does it too. Lu et al. (2019) show that the probability of a whole network being *born dead* goes to 1 as depth grows, and to 0 as width grows at fixed depth.

Mitigations, roughly in order of how often you'll reach for them:

| Fix | Mechanism |
|---|---|
| He initialization | Helps preserve activation variance at initialization; with roughly symmetric pre-activations, many units begin active |
| Lower LR / gradient clipping | Prevents the single oversized update that pushes weights or bias far negative |
| BatchNorm before activation | Re-centers `z` over each batch, so it's hard for a unit to end up negative on every input |
| Leaky ReLU, `max(0.01x, x)` | Gradient is `0.01` for `x < 0`, never `0`, so there's always a way back |
| GELU / SiLU | Smooth transitions that can pass gradient for many negative inputs |

GELU is covered in more depth in [transformer/norm.md](../transformer/norm.md#q-relu-or-gelu-inside-the-ffn).

______________________________________________________________________

## The `x = 0` Point

ReLU is non-differentiable exactly at zero. PyTorch chooses a subgradient of `0` there. This convention matters if an activation is exactly zero, but it does not change the distinction between a single gated example and a unit inactive across the data. See [grad_and_descent.md](grad_and_descent.md#case-2-differentiable-almost-everywhere-kinks) for the PyTorch check.

______________________________________________________________________

## Common confusions

- An inactive ReLU blocks one term in a sum of gradient paths; it does not automatically block the whole layer.
- A unit inactive on today's batch may activate on another batch. Persistent inactivity across the data is the concern.
- Leaky ReLU supplies a negative-side slope; it does not guarantee that every layer's gradients are well behaved.

## Check your understanding

1. If one of three ReLU units has a negative pre-activation, can the input below still receive a gradient?
1. Why do a persistently inactive unit's own incoming weights receive zero task gradient?
1. How could a unit in a deeper network become active again despite that zero incoming-weight gradient?

<details markdown="1"><summary>Answers</summary>

1. Yes. The active units' paths still contribute to the sum.
1. Its local ReLU derivative is zero for every example it sees, so the chain rule multiplies every such contribution by zero.
1. Earlier shared layers can change its input representation and move its pre-activation above zero.

</details>

**One-minute recap:** ReLU gates individual paths; repeated inactivity can stall a unit's own weights; initialization, learning rate, normalization, and activation choice affect how often that happens.

______________________________________________________________________

## References

- Nair & Hinton, *"Rectified Linear Units Improve Restricted Boltzmann Machines"* (ICML 2010). Introduces rectified linear units, in RBMs.
- Glorot, Bordes & Bengio, *"Deep Sparse Rectifier Neural Networks"* (AISTATS 2011). The case for rectifiers in deep networks, including sparse representations with true zeros.
- Maas, Hannun & Ng, *"Rectifier Nonlinearities Improve Neural Network Acoustic Models"* (2013). Leaky ReLU.
- He et al., *"Delving Deep into Rectifiers"* (2015). He initialization and PReLU. arXiv:1502.01852
- Lu et al., *"Dying ReLU and Initialization: Theory and Numerical Examples"* (2019). Formal treatment of born-dead networks and how the probability scales with depth and width. arXiv:1903.06733
- Stanford CS231n notes, *"Neural Networks Part 1: Setting up the Architecture"*. The standard practitioner framing of dying ReLU. cs231n.github.io/neural-networks-1/
