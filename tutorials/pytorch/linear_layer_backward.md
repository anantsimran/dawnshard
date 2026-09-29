# Where the Transpose Comes From

**Remember the cause, not just the formula:** one weight is reused across every example, so its gradient sums over the batch. A transpose puts that batch index in the inner slot where matrix multiplication can perform the sum.

Read the [concrete numbers](#concrete-numbers) once, then use the [shape shortcut](#the-shortcut-everyone-actually-uses) for revision.

Take a linear layer `Y = X W` with a batch of inputs `X` (`n×d`), weights `W` (`d×k`), and output `Y` (`n×k`). Write `G = ∂L/∂Y` (`n×k`) for the gradient arriving from above. Backprop gives:

```
∂L/∂W = Xᵀ G
∂L/∂X = G Wᵀ
```

Why the transposes?

______________________________________________________________________

## One-Line Answer

The transpose isn't a rule you memorize. It's what happens when **the index you need to sum over is sitting in the wrong slot** for matrix multiplication.

Matmul has exactly one behavior: it sums over the *inner* index (columns of the left operand, rows of the right). So if your derivative sums over some index, you have to flip whichever matrix isn't storing that index in the inner position. That flip is the transpose.

______________________________________________________________________

## Where the Sum Comes From

Forward pass, written per-element:

```
Y[i,j] = Σ_a X[i,a] · W[a,j]
```

Now ask: **how many places does a single weight `W[a,j]` show up?** It's used by *every row* of the batch. All `n` examples multiply by the same weight, so its gradient collects a contribution from each one:

```
∂L/∂W[a,j] = Σ_i G[i,j] · X[i,a]        ← sum over i, the batch index
```

That sum runs over `i`. For a matmul to produce it, `i` must be the column index of the left operand. But in `X`, `i` is the **row** index. Flip it to `Xᵀ[a,i]`, and now:

```
Σ_i Xᵀ[a,i] · G[i,j]  =  (Xᵀ G)[a,j]  ✓
```

**Intuition:** the forward pass reads `X` *row by row* (one row = one example). The weight gradient needs `X` *column by column* (one column = one input feature, gathered across all examples). The transpose switches from the "per-example" view to the "per-feature" view.

______________________________________________________________________

## Concrete Numbers

```
X = [[1, 2],     W = [[10],     G = ∂L/∂Y = [[1],
     [3, 4]]          [20]]                  [2]]
n=2, d=2         d=2, k=1       n=2, k=1
```

Expand the forward pass by hand:

```
Y[0] = 1·w₀ + 2·w₁
Y[1] = 3·w₀ + 4·w₁
```

`w₀` appears in both rows, so the chain rule sums both:

```
∂L/∂w₀ = g₀·1 + g₁·3 = 1·1 + 2·3 = 7      ← 1 and 3 are X's first COLUMN
∂L/∂w₁ = g₀·2 + g₁·4 = 1·2 + 2·4 = 10     ← 2 and 4 are X's second COLUMN
```

And indeed `Xᵀ G = [[1,3],[2,4]] @ [[1],[2]] = [[7],[10]]`. ✓

The multipliers you needed were **columns of `X`**. `Xᵀ` is just "`X` with its columns turned into rows so a matmul can dot them."

______________________________________________________________________

## The Other One: `∂L/∂X = G Wᵀ`

Same story, different shared index. A single activation `X[i,a]` feeds *all `k` outputs* in its row, so its gradient sums over `j`:

```
∂L/∂X[i,a] = Σ_j G[i,j] · W[a,j]
```

Here `j` is already the column index of `G`, which is good because `G` is the left operand. But `j` is also the column index of `W`, and the right operand needs it as a *row*. Flip it: `Wᵀ`.

A deeper intuition worth holding onto: **forward, `W` maps `d → k`. Backward, gradient flows `k → d`.** You need the same weights with the arrows reversed, and reversing the arrows of a linear map *is* the transpose (its adjoint). Every backward pass through a linear layer runs the same wiring in reverse.

______________________________________________________________________

## The Shortcut Everyone Actually Uses

Once you trust the derivation, don't redo it. Just match shapes:

| Need | Have | Only option |
|---|---|---|
| `∂L/∂W` is `d×k` | `X` is `n×d`, `G` is `n×k` | `Xᵀ (d×n) @ G (n×k)` |
| `∂L/∂X` is `n×d` | `G` is `n×k`, `W` is `d×k` | `G (n×k) @ Wᵀ (k×d)` |

The gradient of a scalar loss with respect to a tensor always has the same shape as that tensor. Given that, dimensional analysis forces the transposes, as long as `n`, `d` and `k` are all different. With square shapes several arrangements type-check, so check with distinct sizes.

______________________________________________________________________

## In PyTorch: `nn.Linear` Stores the Transpose

`nn.Linear` keeps its weight as `(out, in)`, which is `k×d`, and computes `y = x Wᵀ + b` (see [nn_module_and_multi_layer_networks.md](nn_module_and_multi_layer_networks.md#why-you-need-nonlinearity)). Its `weight` is the transpose of the `W` above, so the same gradients come out rearranged:

```python
lin = nn.Linear(in_features=3, out_features=2)
x = torch.randn(4, 3, requires_grad=True)
g = torch.randn(4, 2)  # stands in for ∂L/∂y
lin(x).backward(gradient=g)

lin.weight.shape                                   # torch.Size([2, 3])  (out, in)
torch.allclose(lin.weight.grad, g.T @ x.detach())  # True: Gᵀ X, i.e. (Xᵀ G)ᵀ
torch.allclose(x.grad, g @ lin.weight.detach())    # True: G W, the transpose is already stored
torch.allclose(lin.bias.grad, g.sum(dim=0))        # True: shared across the batch, so summed over it
```

Same derivation, same shape-matching shortcut. Only the storage layout differs.

______________________________________________________________________

## Common confusions

- `XᵀG` comes from the convention `Y=XW`. `nn.Linear` stores weights as `(out, in)` and uses `XWᵀ`, so its stored weight gradient is `GᵀX`.
- The transpose changes which index matmul sums over. It does not mean the forward pass literally switched rows and columns in memory.
- A bias shared across `n` examples receives a sum of `n` contributions; a mean-reduced loss may already scale each contribution by `1/n`.

## Check your understanding

1. If `X` is `(5, 3)` and `G` is `(5, 2)`, what shape must `∂L/∂W` have for `Y=XW`?
1. Which index is summed in `∂L/∂W[a,j]`?
1. Why is `∂L/∂X = GWᵀ`?

<details markdown="1"><summary>Answers</summary>

1. `(3, 2)`, the shape of `W`; `XᵀG` has that shape.
1. The batch index `i`, because the same weight is used for every example.
1. Each input feature contributes to all output features, so the output index `j` is summed; `Wᵀ` puts `j` in matmul's inner position.

</details>

**One-minute recap:** forward sums over input features; weight backward sums over examples; input backward sums over output features. The required sum determines each transpose.

______________________________________________________________________

## References

- Stanford CS231n notes, *"Backpropagation, Intuitions"*. The matrix-multiply gradient (written `D = W X`, so `dW = dD.dot(X.T)`) and dimension analysis as the practical shortcut. cs231n.github.io/optimization-2/
- Goodfellow, Bengio & Courville, *Deep Learning* (2016), §6.5 "Back-Propagation and Other Differentiation Algorithms". Index-notation treatment of the chain rule over tensors.
- Petersen & Pedersen, *The Matrix Cookbook*, §2.4. Standard matrix derivative identities.
