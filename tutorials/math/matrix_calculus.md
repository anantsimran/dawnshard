# The Matrix Calculus You Need for Deep Learning

> **Adapted from Parr & Howard.** This page is a rewrite of Terence Parr and Jeremy Howard's [*The Matrix Calculus You Need For Deep Learning*](https://explained.ai/matrix-calculus/) (2018, [arXiv:1802.01528](https://arxiv.org/abs/1802.01528)). It keeps the article's section order, examples, and notation, but the text is written for this site and every section adds a PyTorch check. It isn't a copy. Section numbers match the original, so you can read the two side by side. Credit for the approach and the examples belongs to the authors.

**Notation on this page.** In prose, **bold** letters are vectors (**x**, **w**) and plain letters are scalars (`x`, `z`). Code blocks can't show bold, so their comments give each object's shape. Vectors are columns unless marked with a transpose `ᵀ`. `ones(n)` is a column vector of `n` ones.

______________________________________________________________________

## 1. Introduction

Training a network means following the gradient of a loss with respect to every weight and bias. For one neuron, the forward pass is:

```
z = w·x + b               # w, x: (n,)   b, z: scalar
activation = max(0, z)    # ReLU
```

The loss compares the activations with the targets across all `N` training inputs. For mean squared error:

```
C = (1/N) Σᵢ (targetᵢ − max(0, w·xᵢ + b))²
```

You could differentiate that by hand. A real network has many neurons in many layers, though, and every layer takes a vector in and returns a vector. You need two kinds of general rule: one for the derivative of a scalar with respect to a vector, and one for the derivative of a vector with respect to a vector. That area of math is **matrix calculus**. Deep learning only uses a small part of it.

You don't need any of this to train models, because autograd computes the derivatives for you. It matters when you want to understand what autograd is doing, or follow the math in a paper.

______________________________________________________________________

## 2. Review: Scalar Derivative Rules

| Rule | `f(x)` | `d/dx` of it | Example |
|---|---|---|---|
| Constant | `c` | `0` | `d/dx 99 = 0` |
| Multiplication by constant | `c·f` | `c · df/dx` | `d/dx 3x = 3` |
| Power | `xⁿ` | `n·xⁿ⁻¹` | `d/dx x³ = 3x²` |
| Sum | `f + g` | `df/dx + dg/dx` | `d/dx (x² + 3x) = 2x + 3` |
| Difference | `f − g` | `df/dx − dg/dx` | `d/dx (x² − 3x) = 2x − 3` |
| Product | `f·g` | `f · dg/dx + df/dx · g` | `d/dx (x²·x) = x² + 2x·x = 3x²` |
| Chain | `f(g(x))` | `df/du · du/dx`, with `u = g(x)` | `d/dx ln(x²) = (1/x²)·2x = 2/x` |

**Write `d/dx`, not `f′`.** The prime doesn't say which variable you're differentiating with respect to. Once a function has several inputs, that's the first thing that goes wrong.

It helps to treat `d/dx` as an operator: a function goes in and its derivative comes out. Constants factor out of an operator, and it distributes over sums, so a derivative turns into small pieces you already know:

```
d/dx 9(x + x²) = 9 · (d/dx x + d/dx x²) = 9(1 + 2x) = 9 + 18x
```

______________________________________________________________________

## 3. Introduction to Vector Calculus and Partial Derivatives

A layer is a function of many inputs, not one. Take `f(x, y) = 3x²y`. How it changes depends on which input you change, so there's one derivative per input. Each of these is a **partial derivative**, written `∂/∂x`.

To take a partial derivative, differentiate with respect to one variable and **treat every other variable as a constant**:

```
∂/∂x 3x²y = 3y · ∂/∂x x² = 3y · 2x = 6yx       # 3, y are constants here
∂/∂y 3x²y = 3x² · ∂/∂y y = 3x² · 1 = 3x²       # 3, x² are constants here
```

Put the partials in a horizontal vector and you have the **gradient**:

```
∇f(x, y) = [∂f/∂x, ∂f/∂y] = [6yx, 3x²]
```

A gradient belongs to a function that maps several scalars to one scalar. At `(x, y) = (1, 2)`, it's `[12, 3]`:

```python
import torch

v = torch.tensor([1.0, 2.0], requires_grad=True)   # v = (x, y)
(3 * v[0] ** 2 * v[1]).backward()
v.grad   # tensor([12.,  3.])
```

______________________________________________________________________

## 4. Matrix Calculus

Now take a second function of the same two inputs, `g(x, y) = 2x + y⁸`:

```
∂g/∂x = 2 · ∂x/∂x + ∂y⁸/∂x = 2 + 0 = 2
∂g/∂y = ∂2x/∂y + ∂y⁸/∂y = 0 + 8y⁷ = 8y⁷

∇g(x, y) = [2, 8y⁷]
```

Two functions give two gradients. Stack them as rows and you get the **Jacobian matrix**:

```
    [ ∇f ]   [ ∂f/∂x  ∂f/∂y ]   [ 6yx  3x²  ]
J = [ ∇g ] = [ ∂g/∂x  ∂g/∂y ] = [ 2    8y⁷  ]
```

**There are two layout conventions.** This page uses **numerator layout**, where each function gets a row and each input gets a column. Many papers and libraries use **denominator layout**, which is the transpose:

```
[ 6yx  2   ]
[ 3x²  8y⁷ ]
```

If the shapes in a derivation don't line up, check which layout it uses before you hunt for an algebra mistake.

`torch.autograd.functional.jacobian` uses numerator layout:

```python
from torch.autograd.functional import jacobian

def fg(v):
    x, y = v
    return torch.stack([3 * x ** 2 * y, 2 * x + y ** 8])

J = jacobian(fg, torch.tensor([1.0, 2.0]))
J     # tensor([[  12.,    3.],
      #         [   2., 1024.]])
J.T   # denominator layout
```

### 4.1 Generalization of the Jacobian

Pack the inputs into one column vector **x** of length `n = |x|`, whose elements are the scalars `xᵢ`. Do the same with the outputs: **y** = **f**(**x**) is a stack of `m` scalar functions, and each one reads all of **x**:

```
y₁ = f₁(x)
y₂ = f₂(x)          # x: (n,)   y: (m,)   each fᵢ returns a scalar
 ⋮
yₘ = fₘ(x)
```

With `x₁` in place of `x` and `x₂` in place of `y`, the example from §4 becomes:

```
y₁ = f₁(x) = 3x₁²x₂
y₂ = f₂(x) = 2x₁ + x₂⁸
```

The Jacobian holds every partial derivative of every output with respect to every input. **Row `i` is output `i`, and column `j` is input `j`**:

```
        [ ∇f₁ ]   [ ∂f₁/∂x₁  ∂f₁/∂x₂  …  ∂f₁/∂xₙ ]
∂y      [ ∇f₂ ]   [ ∂f₂/∂x₁  ∂f₂/∂x₂  …  ∂f₂/∂xₙ ]
──  =   [  ⋮  ] = [    ⋮        ⋮     ⋱     ⋮    ]      m × n
∂x      [ ∇fₘ ]   [ ∂fₘ/∂x₁  ∂fₘ/∂x₂  …  ∂fₘ/∂xₙ ]
```

- **By row:** row `i` is the gradient of `fᵢ`, a horizontal `n`-vector.
- **By column:** column `j` shows how every output changes when you nudge `xⱼ` alone.

The width is always the number of inputs, one column for each input you can change. The height is always the number of outputs. Every combination of scalar and vector fits that rule:

| | scalar input `x` | vector input **x** (n) |
|---|---|---|
| **scalar output** `f` | `∂f/∂x`: scalar | `∂f/∂x`: `1 × n` (row) |
| **vector output** **f** (m) | `∂f/∂x`: `m × 1` (column) | `∂f/∂x`: `m × n` |

**The identity function**, **y** = **x**, has `m = n`, so its Jacobian is square. Output `i` is just `xᵢ`, and `xᵢ` doesn't change when you nudge a different element `xⱼ`:

```
∂xᵢ/∂xⱼ = 1   if i = j
        = 0   if i ≠ j

          [ 1  0  …  0 ]
∂y/∂x  =  [ 0  1  …  0 ]  =  I
          [       ⋱    ]
          [ 0  0  …  1 ]
```

That derivation shows a trick worth keeping. **When a vector derivative is confusing, write out one scalar element, take its partial derivatives with the ordinary rules, and then put the results back into a vector or matrix.**

Keep two distinctions in mind throughout. A vector can be vertical (**x**) or horizontal (**x**ᵀ), and a result can be one scalar function (`y = …`) or a vector of functions (**y** = …).

```python
jacobian(lambda x: x, torch.ones(3))
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

jacobian(lambda s: s ** 2, torch.tensor(3.0)).shape                     # ()      scalar → scalar
jacobian(lambda v: v.sum(), torch.ones(3)).shape                        # (3,)    vector → scalar
jacobian(lambda s: torch.stack([s, s ** 2]), torch.tensor(3.0)).shape   # (2,)    scalar → vector
jacobian(lambda v: torch.stack([v.sum(), v.prod()]), torch.ones(3)).shape  # (2, 3)  vector → vector
```

PyTorch drops size-1 axes, so a `1 × n` row comes back as `(n,)`.

### 4.2 Derivatives of Vector Element-wise Binary Operators

An **element-wise binary operator** combines two vectors position by position: element 1 with element 1, element 2 with element 2, and so on. Examples include **w** + **x**, **w** > **x**, and max(**w**, **x**). NumPy and PyTorch arithmetic behave this way by default.

Write the general case as **y** = **f**(**w**) ○ **g**(**x**), where ○ is any such operator. (○ here isn't function composition.) Every vector has the same length `n`:

```
[ y₁ ]   [ f₁(w) ○ g₁(x) ]
[ y₂ ] = [ f₂(w) ○ g₂(x) ]      # w, x, y: (n,)
[ ⋮  ]   [       ⋮       ]
[ yₙ ]   [ fₙ(w) ○ gₙ(x) ]
```

The Jacobian with respect to **w** is an `n × n` matrix, and entry `(i, j)` is `∂/∂wⱼ (fᵢ(w) ○ gᵢ(x))`. The Jacobian with respect to **x** has the same form. In general, every entry can be nonzero.

**It's usually diagonal.** Suppose `fᵢ` reads only `wᵢ` and `gᵢ` reads only `xᵢ`. This is the **element-wise diagonal condition**, and **w** + **x** satisfies it because output `i` is `wᵢ + xᵢ`. Then for `j ≠ i`, output `i` doesn't depend on `wⱼ` at all. It's a constant with respect to `wⱼ`, so its partial derivative is 0. Only the diagonal is left:

```
∂y/∂w = diag( ∂/∂w₁ (f₁(w₁) ○ g₁(x₁)),  …,  ∂/∂wₙ (fₙ(wₙ) ○ gₙ(xₙ)) )
∂y/∂x = diag( ∂/∂x₁ (f₁(w₁) ○ g₁(x₁)),  …,  ∂/∂xₙ (fₙ(wₙ) ○ gₙ(xₙ)) )
```

`diag(v)` builds a square matrix with `v` on the diagonal and zeros everywhere else. (Writing `fᵢ(wᵢ)` stretches the notation, since `fᵢ` takes a vector, but the meaning is clear.)

In plain vector arithmetic, **f**(**w**) is often just **w**, so `fᵢ(wᵢ) = wᵢ`. Each diagonal entry is then a one-variable derivative. For addition, `∂(wᵢ + xᵢ)/∂wᵢ = 1`, which gives the identity matrix. Here are the four arithmetic operators, where ⊗ is element-wise multiplication (the Hadamard product) and ⊘ is element-wise division:

| Op | `∂/∂w` | `∂/∂x` |
|---|---|---|
| **w** + **x** | `diag(ones(n)) = I` | `I` |
| **w** − **x** | `I` | `diag(−ones(n)) = −I` |
| **w** ⊗ **x** | `diag(x)` | `diag(w)` |
| **w** ⊘ **x** | `diag(…, 1/xᵢ, …)` | `diag(…, −wᵢ/xᵢ², …)` |

```python
w = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([4.0, 5.0, 6.0])

jacobian(lambda w: w * x, w)
# tensor([[4., 0., 0.],
#         [0., 5., 0.],
#         [0., 0., 6.]])     ← diag(x)

torch.allclose(jacobian(lambda x: w / x, x), torch.diag(-w / x ** 2))   # True
torch.allclose(jacobian(lambda x: w - x, x), -torch.eye(3))             # True
```

### 4.3 Derivatives Involving Scalar Expansion

Adding a scalar to a vector, or multiplying a vector by a scalar, is an element-wise operation in disguise. The scalar is first expanded into a vector:

```
y = x + z   is   y = f(x) + g(z),   f(x) = x,   g(z) = ones(n)·z      # x: (n,)   z: scalar
y = x z     is   y = x ⊗ ones(n)·z
```

That expansion is broadcasting (see [Shapes & broadcasting](../pytorch/broadcasting.md)). Here, `z` doesn't depend on **x**, so `∂z/∂xᵢ = 0`. Both expansions meet the diagonal condition, so the §4.2 rule applies:

```
∂(x + z)/∂x:  diagonal entries  ∂(xᵢ + z)/∂xᵢ = 1          →  I
∂(x z)/∂x:    diagonal entries  ∂(xᵢ z)/∂xᵢ  = z          →  I z     (product rule: xᵢ·0 + z·1)
```

With respect to the scalar `z`, the result is a column vector, not a diagonal matrix. `z` is a single input that every output reads:

```
∂(x + z)/∂z:  entries  ∂(xᵢ + z)/∂z = 1     →  ones(n)
∂(x z)/∂z:    entries  ∂(xᵢ z)/∂z  = xᵢ    →  x
```

In backprop, every output's gradient then flows back into the one shared scalar and gets added up. That's why a bias gradient is a sum, as §6.2 shows.

```python
z = torch.tensor(3.0)
jacobian(lambda z: x + z, z)   # tensor([1., 1., 1.])
jacobian(lambda z: x * z, z)   # tensor([4., 5., 6.])     ← x

jacobian(lambda x: x * z, x)
# tensor([[3., 0., 0.],
#         [0., 3., 0.],
#         [0., 0., 3.]])     ← I z
```

### 4.4 Vector Sum Reduction

Summing a vector's elements shows up in loss functions. It also makes other vector-to-scalar operations, such as the dot product, easy to differentiate.

Let `y = sum(f(x)) = Σᵢ fᵢ(x)`. The sum runs over the function's **outputs**, and each `fᵢ` may read all of **x**. The result is a scalar, so its Jacobian is a `1 × n` row. The derivative moves inside the sum:

```
∂y/∂x = [ ∂y/∂x₁,        ∂y/∂x₂,        …,  ∂y/∂xₙ        ]
      = [ Σᵢ ∂fᵢ/∂x₁,    Σᵢ ∂fᵢ/∂x₂,    …,  Σᵢ ∂fᵢ/∂xₙ    ]
```

**`y = sum(x)`**, so `fᵢ(x) = xᵢ`. In column `j`, every term except `i = j` is zero:

```
∇y = [ ∂x₁/∂x₁, ∂x₂/∂x₂, …, ∂xₙ/∂xₙ ] = [1, 1, …, 1] = ones(n)ᵀ
```

The result is a **row** of ones. Getting that orientation right is what keeps the shapes consistent in larger derivations.

**`y = sum(x z)`**, so `fᵢ(x, z) = xᵢ z`:

```
∂y/∂x = [ ∂(x₁z)/∂x₁, …, ∂(xₙz)/∂xₙ ] = [z, z, …, z]
∂y/∂z = Σᵢ ∂(xᵢ z)/∂z = Σᵢ xᵢ = sum(x)                  # 1 × 1
```

```python
jacobian(lambda x: x.sum(), x)          # tensor([1., 1., 1.])
jacobian(lambda x: (x * z).sum(), x)    # tensor([3., 3., 3.])
jacobian(lambda z: (x * z).sum(), z)    # tensor(15.)   ← sum(x) = 4 + 5 + 6
```

### 4.5 The Chain Rules

The rules so far don't cover nested expressions like `sum(w + x)`, unless you break them down into scalars by hand. Combining the rules takes a chain rule. Several different rules share that name, so this section names three of them and says when each one applies:

1. The **single-variable chain rule**: a scalar function of a scalar, through a single path.
1. The **single-variable total-derivative chain rule**: a scalar function of a scalar, where the input reaches the output through more than one path.
1. The **vector chain rule**, which covers both of the above and everything a network needs.

All three work by divide and conquer. You split an expression into small subexpressions, differentiate each one on its own, and combine the pieces. That's also how autograd works.

#### 4.5.1 Single-Variable Chain Rule

`d/dx sin(x²) = 2x·cos(x²)`. The answer is the derivative of the outer part, `cos(u)`, times the derivative of the inner part, `2x`. To keep track of the variable you're differentiating with respect to, name the inner result explicitly:

```
dy/dx = dy/du · du/dx
```

The process has four steps:

1. **Introduce intermediate variables.** Give each subexpression its own variable, so each variable is defined by a single operator or function call. Working from the innermost subexpression outward means each variable depends only on variables you've already defined.
1. **Differentiate each intermediate** with respect to its own parameter.
1. **Multiply** the derivatives together. This is the "chain" step.
1. **Substitute** the original expressions back in.

For `y = sin(x²)`:

```
1.  u = x²                 y = sin(u)
2.  du/dx = 2x             dy/du = cos(u)       ← with respect to u, not x
3.  dy/dx = cos(u) · 2x
4.  dy/dx = 2x·cos(x²)
```

As a dataflow graph, this is `x → square → sin → y`. A change in `x` reaches `y` along **exactly one path**, and that's the condition for using this rule. A simpler test is slightly stricter: every intermediate function takes exactly one parameter. `y = x + x²` fails the test, because after `u = x²`, `y = x + u` has two parameters, and `x` reaches `y` along two paths. That case needs §4.5.2.

**Forward vs. backward.** The factors multiply in either order. The order corresponds to the direction you move through the graph:

| Forward differentiation (x to y) | Backward differentiation (y to x) |
|---|---|
| `dy/dx = du/dx · dy/du` | `dy/dx = dy/du · du/dx` |

Forward differentiation measures how one input affects the output, so it needs one pass per input. Backward differentiation starts at the output and reaches every input in one pass. A network has one scalar loss and millions of parameters, so backward differentiation is far cheaper. `.backward()` is backward differentiation.

**A longer chain.** For `y = ln(sin(x³)²)`, compute the intermediates in order, the way a compiler evaluates nested calls:

```
1.  u₁ = x³        u₂ = sin(u₁)        u₃ = u₂²        u₄ = ln(u₃)      (y = u₄)
2.  du₁/dx = 3x²   du₂/du₁ = cos(u₁)   du₃/du₂ = 2u₂   du₄/du₃ = 1/u₃
3.  dy/dx = (1/u₃) · 2u₂ · cos(u₁) · 3x² = 6u₂x²cos(u₁) / u₃
4.  dy/dx = 6 sin(x³) x² cos(x³) / sin(x³)²  =  6x²cos(x³) / sin(x³)
```

```python
x1 = torch.tensor(1.0, requires_grad=True)
torch.sin(x1 ** 2).backward()
x1.grad                                  # tensor(1.0806)   = 2·cos(1)

x1 = torch.tensor(1.0, requires_grad=True)
torch.log(torch.sin(x1 ** 3) ** 2).backward()
x1.grad                                  # tensor(3.8526)   = 6·cos(1)/sin(1)
```

#### 4.5.2 Single-Variable Total-Derivative Chain Rule

Take `y = x + x²`. The answer is `1 + 2x`, but the §4.5.1 rule gets it wrong:

```
u₁ = x²
u₂ = x + u₁          (y = u₂)

"du₂/du₁ · du₁/dx" = 1 · 2x = 2x        ✗  misses the direct x → u₂ path
```

Using partial derivatives doesn't fix it either. `∂u₂/∂x = 1` holds `u₁` constant, but `u₁` depends on `x`, so it changes whenever `x` does. Try numbers. At `x = 1`, `y = 2`. At `x = 2`, `y = 2 + 4 = 6`. Increasing `x` by 1 raised `y` by 4, not 1, because `x` changes `y` along two paths: directly through the `+`, and indirectly through the square.

The **total derivative** adds up the contributions from every path:

```
dy/dx = ∂u₂/∂x  +  ∂u₂/∂u₁ · ∂u₁/∂x  =  1 + 1·2x  =  1 + 2x
         direct      through u₁
```

In general, for `f(x, u₁, …, uₙ)` where any of the `uᵢ` may depend on `x`:

```
∂f/∂x = ∂f/∂x + Σᵢ ∂f/∂uᵢ · ∂uᵢ/∂x
```

The left-hand side is the total derivative, even though it's written with `∂`. On the right, the first `∂f/∂x` is the ordinary partial, which holds every `uᵢ` fixed. A partial derivative holds everything else constant, while a total derivative lets everything that depends on `x` change. This page follows the original article and writes both with `∂`, to match the vector rule in §4.5.3. Wikipedia uses `d` for the total derivative.

**A nested example**, `f(x) = sin(x + x²)`:

```
u₁ = x²             ∂u₁/∂x = 2x
u₂ = x + u₁         ∂u₂/∂x = 1 + 1·2x = 1 + 2x
u₃ = sin(u₂)        ∂u₃/∂x = 0 + cos(u₂)·∂u₂/∂x = cos(x + x²)(1 + 2x)
```

**The rule always adds.** It isn't adding because `x + x²` contains a `+`. It adds because a change in `x` spreads along several paths, and each path contributes separately. Use multiplication for the second operation, `y = x · x²`, and leave it unsimplified:

```
u₁ = x²             ∂u₁/∂x  = 2x
u₂ = x·u₁           ∂u₂/∂x  = u₁         ∂u₂/∂u₁ = x

dy/dx = ∂u₂/∂x + ∂u₂/∂u₁ · ∂u₁/∂x = x² + x·2x = 3x²
```

The formula didn't change. Only the partial derivatives inside it did.

Autograd applies this rule automatically. When a tensor is used more than once, the contributions from each use are **added** into its `.grad`:

```python
x1 = torch.tensor(1.0, requires_grad=True)
(x1 + x1 ** 2).backward()
x1.grad                  # tensor(3.)        = 1 + 2·1

x1 = torch.tensor(1.0, requires_grad=True)
torch.sin(x1 + x1 ** 2).backward()
x1.grad                  # tensor(-1.2484)   = cos(2)·3

x1 = torch.tensor(2.0, requires_grad=True)
(x1 * x1 ** 2).backward()
x1.grad                  # tensor(12.)       = 3·2²
```

**Why introduce intermediates even for `x²`?** It keeps each derivative trivial, it lets the rule take a simpler form, and it's what autograd does.

**The final form.** Treat `x` as one more intermediate, `uₙ₊₁ = x`. The extra term disappears into the sum:

```
∂f(u₁, …, uₙ₊₁)/∂x = Σᵢ ∂f/∂uᵢ · ∂uᵢ/∂x
```

With a single path, the sum has one term, and this reduces to the §4.5.1 rule, so this is the one to remember. The sum is also a dot product, `∂f/∂u · ∂u/∂x`, which leads into §4.5.3.

**Watch the terminology.** Most calculus material calls this rule the "multivariable chain rule." `f(x) = x + x²` is still a scalar function of one scalar, though. Only its intermediates take more than one variable. In a matrix-calculus context, the name is misleading.

#### 4.5.3 Vector Chain Rule

Rather than state the vector rule outright, derive it from an example. Start with a vector function of a scalar:

```
y = f(x) = [ ln(x²)  ]      # x: scalar   y: (2,)
           [ sin(3x) ]
```

Introduce one intermediate for each element, so that **y** = **f**(**g**(x)):

```
g(x) = [ x² ]         f(g) = [ ln(g₁)  ]
       [ 3x ]                [ sin(g₂) ]
```

Apply the total-derivative rule to each element. Each `fᵢ` could read either `g`:

```
∂y/∂x = [ ∂f₁/∂g₁·∂g₁/∂x + ∂f₁/∂g₂·∂g₂/∂x ]   [ (1/g₁)·2x + 0  ]   [ 2/x      ]
        [ ∂f₂/∂g₁·∂g₁/∂x + ∂f₂/∂g₂·∂g₂/∂x ] = [ 0 + cos(g₂)·3 ] = [ 3cos(3x) ]
```

Every term has the form `∂fᵢ/∂gⱼ · ∂gⱼ/∂x`. Factor the `∂gⱼ/∂x` terms out into their own vector, and what's left is a matrix multiplied by a vector:

```
[ ∂f₁/∂g₁  ∂f₁/∂g₂ ] [ ∂g₁/∂x ]     ∂f   ∂g
[ ∂f₂/∂g₁  ∂f₂/∂g₂ ] [ ∂g₂/∂x ]  =  ── · ──
                                    ∂g   ∂x
```

The Jacobian of the composition is **the product of the two Jacobians**. Check it:

```
[ 1/g₁  0       ] [ 2x ]   [ 2x/g₁      ]   [ 2/x      ]
[ 0     cos(g₂) ] [ 3  ] = [ 3·cos(g₂)  ] = [ 3cos(3x) ]      ✓
```

This has the same shape as the scalar chain rule, and it still holds when you replace the scalar `x` with a vector **x**:

```
∂/∂x f(g(x)) = ∂f/∂g · ∂g/∂x           (m × k) · (k × n) = m × n
```

Here `m = |f|`, `k = |g|`, and `n = |x|`. Matrix multiplication doesn't commute, so keep the factors in this order. The total derivative is built in: entry `(i, j)` of the product sums over every intermediate `gₖ` that connects `xⱼ` to `fᵢ`.

```
                 [ ∂f₁/∂g₁  …  ∂f₁/∂gₖ ] [ ∂g₁/∂x₁  …  ∂g₁/∂xₙ ]
∂/∂x f(g(x))  =  [    ⋮     ⋱     ⋮    ] [    ⋮     ⋱     ⋮    ]
                 [ ∂fₘ/∂g₁  …  ∂fₘ/∂gₖ ] [ ∂gₖ/∂x₁  …  ∂gₖ/∂xₙ ]
```

**Element-wise layers make it diagonal.** Neural networks mostly apply functions to vectors element by element, for example `max(0, x)`. If `fᵢ` reads only `gᵢ` and `gᵢ` reads only `xᵢ`, both Jacobians are diagonal, and multiplying them just multiplies the diagonals:

```
∂/∂x f(g(x)) = diag(∂fᵢ/∂gᵢ) · diag(∂gᵢ/∂xᵢ) = diag(∂fᵢ/∂gᵢ · ∂gᵢ/∂xᵢ)
```

Each diagonal entry is a single-variable chain rule.

**You only need the vector chain rule.** The two scalar rules are special cases of it. This table gives the shape of each factor for every combination of scalar and vector:

| `∂f/∂u · ∂u/∂x` | `x` scalar, `u` scalar | `x` scalar, **u** vector (k) | **x** vector (n), **u** vector (k) |
|---|---|---|---|
| **`f` scalar** | `1×1 · 1×1` | `1×k · k×1` | `1×k · k×n` |
| **f vector (m)** | `m×1 · 1×1` | `m×k · k×1` | `m×k · k×n` |

```python
s = torch.tensor(2.0)
jacobian(lambda s: torch.stack([torch.log(s ** 2), torch.sin(3 * s)]), s)
# tensor([1.0000, 2.8805])      ← [2/x, 3cos(3x)] at x = 2

g = torch.stack([s ** 2, 3 * s])
df_dg = torch.diag(torch.stack([1 / g[0], torch.cos(g[1])]))   # 2 × 2
dg_dx = torch.stack([2 * s, torch.tensor(3.0)]).unsqueeze(1)   # 2 × 1
df_dg @ dg_dx
# tensor([[1.0000],
#         [2.8805]])
```

______________________________________________________________________

## 5. The Gradient of Neuron Activation

That covers everything needed to differentiate a neuron with respect to its parameters **w** and `b`:

```
activation(x) = max(0, w·x + b)      # w, x: (n,)   b: scalar
```

Other affine functions, such as convolution, and other activations, such as ELU, follow the same steps.

**The dot product.** Start with `y = w·x`. Its derivative follows from the rules you already have. A dot product is a sum of element-wise products, `w·x = Σᵢ wᵢxᵢ = sum(w ⊗ x)`, which is also `wᵀx`:

```
u = w ⊗ x            ∂u/∂w = diag(x)       (§4.2)
y = sum(u)           ∂y/∂u = ones(n)ᵀ      (§4.4)

∂y/∂w = ∂y/∂u · ∂u/∂w = ones(n)ᵀ · diag(x) = xᵀ
```

Check one element with scalars. Only the `i = j` term of the sum depends on `wⱼ`:

```
∂/∂wⱼ Σᵢ wᵢxᵢ = ∂/∂wⱼ (wⱼxⱼ) = xⱼ      →   ∂y/∂w = [x₁, …, xₙ] = xᵀ   ✓
```

**Add the bias.** For `y = w·x + b`, the sum rule is enough, with no chain rule needed:

```
∂y/∂w = xᵀ + zeros(n)ᵀ = xᵀ
∂y/∂b = 0 + 1 = 1
```

**The ReLU.** `max(0, z)` has two pieces. For `z ≤ 0` the output is the constant 0, and for `z > 0` it's `z`:

```
d/dz max(0, z) = 0   if z ≤ 0
               = 1   if z > 0
```

(At exactly `z = 0` the derivative isn't defined. This page takes it as 0, and so does PyTorch. See [ReLU & dead neurons](../pytorch/relu_and_dead_neurons.md).) Given a vector, `max(0, x)` applies the scalar function to each element. It's an element-wise unary operator, and its derivative is 0 or 1 at each position.

**Chain them.** Let `z = w·x + b` be the intermediate and `activation = max(0, z)`:

```
∂activation/∂w = ∂activation/∂z · ∂z/∂w
               = 0 · xᵀ = zeros(n)ᵀ     if w·x + b ≤ 0
               = 1 · xᵀ = xᵀ            if w·x + b > 0

∂activation/∂b = 0                      if w·x + b ≤ 0
               = 1                      if w·x + b > 0
```

This matches intuition. When the ReLU clips the output to 0, no small change to a weight changes the output. When it doesn't clip, the ReLU acts like the identity, so the gradient is the affine function's gradient.

```python
w = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

x_on = torch.tensor([3.0, 1.0, 2.0])      # z = 3 − 2 + 1 + 0.5 = 2.5 > 0
torch.relu(w @ x_on + b).backward()
w.grad, b.grad                            # (tensor([3., 1., 2.]), tensor(1.))   ← xᵀ, 1

w.grad, b.grad = None, None
x_off = torch.tensor([0.0, 3.0, 1.0])     # z = 0 − 6 + 0.5 + 0.5 = −5 ≤ 0
torch.relu(w @ x_off + b).backward()
w.grad, b.grad                            # (tensor([0., 0., 0.]), tensor(0.))
```

______________________________________________________________________

## 6. The Gradient of the Neural Network Loss Function

Training uses `N` input vectors with a scalar target for each one:

```
X = [x₁, x₂, …, x_N]ᵀ                      # N × n, one input per row
y = [target(x₁), …, target(x_N)]ᵀ          # (N,)

C(w, b, X, y) = (1/N) Σᵢ (yᵢ − max(0, w·xᵢ + b))²
```

Introduce intermediate variables, as in §4.5. Here `u` is one example's activation and `v` is its error:

```
u(w, b, x) = max(0, w·x + b)
v(y, u)    = y − u
C(v)       = (1/N) Σᵢ vᵢ²
```

### 6.1 The Gradient with Respect to the Weights

From §5, and since `y` doesn't depend on **w**:

```
∂u/∂w = zeros(n)ᵀ  if w·x + b ≤ 0          ∂v/∂w = ∂(y − u)/∂w = −∂u/∂w
      = xᵀ         if w·x + b > 0                = zeros(n)ᵀ  if w·x + b ≤ 0
                                                 = −xᵀ        if w·x + b > 0
```

Move the derivative inside the average and chain through `v²`:

```
∂C/∂w = (1/N) Σᵢ ∂vᵢ²/∂vᵢ · ∂vᵢ/∂w
      = (1/N) Σᵢ 2vᵢ · ∂vᵢ/∂w
      = (1/N) Σᵢ { zeros(n)ᵀ                         if w·xᵢ + b ≤ 0
                 { −2(yᵢ − (w·xᵢ + b)) xᵢᵀ           if w·xᵢ + b > 0
```

In the active case `u = w·xᵢ + b`, so `max` disappears. The condition is checked **per example**: an example whose neuron is clipped contributes nothing to the sum. (The original article writes the case split outside the sum as shorthand.) Define the error `eᵢ = w·xᵢ + b − yᵢ` and flip the sign:

```
∂C/∂w = (2/N) Σ_{i : w·xᵢ + b > 0}  eᵢ xᵢᵀ
```

**Reading it.** The gradient is a weighted sum of the input vectors, weighted by their errors. Inputs with large errors pull hardest. With one example (`N = 1`) the gradient is `2e₁x₁ᵀ`:

- `e₁ = 0`: the gradient is zero, and the loss is at its minimum.
- small `e₁ > 0`: a short step in the direction of **x**₁.
- large `e₁`: a long step in that direction.
- `e₁ < 0`: the direction reverses.

The gradient points toward **higher** loss, so gradient descent steps the other way, with learning rate `η`:

```
w_{t+1} = w_t − η · ∂C/∂w
```

### 6.2 The Derivative with Respect to the Bias

The same intermediates work for `b`:

```
∂u/∂b = 0  if w·x + b ≤ 0            ∂v/∂b = −∂u/∂b = 0   if w·x + b ≤ 0
      = 1  if w·x + b > 0                           = −1  if w·x + b > 0

∂C/∂b = (1/N) Σᵢ 2vᵢ · ∂vᵢ/∂b
      = (2/N) Σ_{i : w·xᵢ + b > 0}  eᵢ
```

The bias gradient adds up the errors of the active examples and scales the total by `2/N`. It's a sum because one `b` is shared by every example, as in §4.3. It gets the same update:

```
b_{t+1} = b_t − η · ∂C/∂b
```

**Folding the bias into the weights.** Append `b` to **w** and a constant 1 to **x**. Then `w·x + b` becomes a single dot product, and one gradient covers both:

```
ŵ = [wᵀ, b]ᵀ      x̂ = [xᵀ, 1]ᵀ      w·x + b = ŵ·x̂
```

Those two partials are everything gradient descent needs for this neuron. Autograd agrees on a random batch:

```python
torch.manual_seed(0)
X = torch.randn(8, 3)                              # N = 8, n = 3
y = torch.randn(8)
w = torch.randn(3, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

C = ((y - torch.relu(X @ w + b)) ** 2).mean()
C.backward()

with torch.no_grad():
    z = X @ w + b
    e = z - y
    active = z > 0
    N = len(y)
    dC_dw = 2 / N * (e[active, None] * X[active]).sum(dim=0)
    dC_db = 2 / N * e[active].sum()

torch.allclose(w.grad, dC_dw), torch.allclose(b.grad, dC_db)   # (True, True)
```

______________________________________________________________________

## 7. Summary

The whole article comes down to a few ideas:

- **Shape first.** A derivative has one row per output and one column per input, as in §4.1.
- **Element-wise operations give diagonal Jacobians** (§4.2, §4.3). Each diagonal entry is an ordinary scalar derivative.
- **Reductions give rows** (§4.4). `∂sum(x)/∂x = ones(n)ᵀ`.
- **One chain rule.** `∂f/∂x = ∂f/∂g · ∂g/∂x` (§4.5.3). It covers the scalar and total-derivative chain rules as special cases.

The next step is derivatives with respect to **matrices**, such as a layer's weight matrix. [Linear-layer backward](../pytorch/linear_layer_backward.md) works through one in this repo.

______________________________________________________________________

## 8. Matrix Calculus Reference

### 8.1 Gradients and Jacobians

```
∇f(x, y) = [∂f/∂x, ∂f/∂y]                           horizontal

        [ ∂f₁/∂x₁  …  ∂f₁/∂xₙ ]
∂y/∂x = [    ⋮     ⋱     ⋮    ]                     m × n,  m = |f|,  n = |x|
        [ ∂fₘ/∂x₁  …  ∂fₘ/∂xₙ ]

∂x/∂x = I
```

### 8.2 Element-wise Operations on Vectors

For **y** = **f**(**w**) ○ **g**(**x**), where `fᵢ` reads only `wᵢ` and `gᵢ` reads only `xᵢ`:

```
∂y/∂w = diag(…, ∂/∂wᵢ (fᵢ(wᵢ) ○ gᵢ(xᵢ)), …)
```

| Op | `∂/∂w` | `∂/∂x` |
|---|---|---|
| **w** + **x** | `I` | `I` |
| **w** − **x** | `I` | `−I` |
| **w** ⊗ **x** | `diag(x)` | `diag(w)` |
| **w** ⊘ **x** | `diag(…, 1/xᵢ, …)` | `diag(…, −wᵢ/xᵢ², …)` |

### 8.3 Scalar Expansion

| Expression | `∂/∂x` | `∂/∂z` |
|---|---|---|
| **x** + `z` | `I` | `ones(n)` |
| **x** `z` | `I z` | **x** |

### 8.4 Vector Reductions

| Expression | Gradient |
|---|---|
| `y = sum(f(x))` | `∂y/∂x = [Σᵢ ∂fᵢ/∂x₁, …, Σᵢ ∂fᵢ/∂xₙ]` |
| `y = sum(x)` | `∂y/∂x = ones(n)ᵀ` |
| `y = sum(x z)` | `∂y/∂x = [z, …, z]`, `∂y/∂z = sum(x)` |
| `y = w·x = sum(w ⊗ x)` | `∂y/∂x = wᵀ`, `∂y/∂w = xᵀ` |

### 8.5 Chain Rules

| Rule | Formula | Use when |
|---|---|---|
| Single-variable | `df/dx = df/du · du/dx` | every intermediate takes one variable (a single path) |
| Single-variable total-derivative | `∂f(u₁, …, uₙ)/∂x = ∂f/∂u · ∂u/∂x` | some intermediates take several variables |
| Vector | `∂/∂x f(g(x)) = ∂f/∂g · ∂g/∂x` | always; the other two are special cases |

______________________________________________________________________

## 9. Notation

| Symbol | Meaning |
|---|---|
| **x** (bold, in prose) | a column vector, `n × 1` |
| `x` | a scalar |
| `xᵢ` | element `i` of **x** (a scalar) |
| `len(x)` | the length of **x**, written between vertical bars in the original |
| `xᵀ` | the transpose of **x** (a row) |
| `Σᵢ₌ₐᵇ xᵢ` | a for loop over `i` from `a` to `b` that adds up the `xᵢ` |
| `I` | the square identity matrix |
| `diag(x)` | a square matrix with **x** on the diagonal and zeros elsewhere |
| `ones(n)`, `zeros(n)` | column vectors of ones and zeros (`1⃗` and `0⃗` in the original) |
| **w** · **x** | the dot product `Σᵢ wᵢxᵢ = sum(w ⊗ x) = wᵀx` |
| ⊗, ⊘ | element-wise multiplication and division |
| `d/dx` | the derivative of a function of one variable |
| `∂/∂x` | the partial derivative, holding all other variables constant |
| `∇f` | the gradient: all partial derivatives of a scalar function, as a row |
| `J`, `∂y/∂x` | the Jacobian: the gradients of several functions, stacked as rows |
| `y = a if cond₁; b if cond₂` | a piecewise definition, written with a brace in the original |

______________________________________________________________________

## 10. Resources

- **The original:** Parr & Howard, [*The Matrix Calculus You Need For Deep Learning*](https://explained.ai/matrix-calculus/), also as a PDF at [arXiv:1802.01528](https://arxiv.org/abs/1802.01528).
- [Matrix calculus on Wikipedia](https://en.wikipedia.org/wiki/Matrix_calculus). A good explanation of numerator vs. denominator layout.
- [matrixcalculus.org](http://www.matrixcalculus.org/). Computes matrix derivatives symbolically.
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf). A dense reference of identities.
- Michael Nielsen, [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/chap1.html). Backpropagation from first principles.
- Dumoulin & Visin, [*A guide to convolution arithmetic for deep learning*](https://arxiv.org/abs/1603.07285).

**Related pages on this site:**

- [Gradients & descent](../pytorch/grad_and_descent.md): a chain rule traced through autograd.
- [Linear-layer backward](../pytorch/linear_layer_backward.md): where the transposes in `∂L/∂W = Xᵀ G` come from.
- [ReLU & dead neurons](../pytorch/relu_and_dead_neurons.md): what happens when the §5 gradient is 0 for every input.
- [Pre-norm vs. post-norm](../transformer/norm.md): gradient flow in a transformer, as a product of layer Jacobians.
