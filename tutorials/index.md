# Tutorials Index

Every tutorial on one page. [Pages](#pages) lists each one with what it covers. [Concepts](#concepts) maps a specific question to the section that answers it.

Folders group pages by topic:

| Folder | Topic |
|---|---|
| `pytorch/` | PyTorch basics: tensors, shapes, autograd, modules, data, debugging |
| `training/` | The training loop, optimizers, schedulers, learning rates, observability |
| `math/` | The calculus under backprop |
| `vision/` | Convolutional networks |
| `transformer/` | Attention, masks, normalization |

**Suggested order** if you're starting from zero: [pytorch/overview.md](pytorch/overview.md), then down the [Pages](#pages) table. Read `math/` alongside `pytorch/grad_and_descent.md` when the shapes in backprop stop making sense.

______________________________________________________________________

## Pages

### `pytorch/`

| Page | What it covers |
|---|---|
| [overview.md](pytorch/overview.md) | What PyTorch is, and the five building blocks of any model |
| [tensors.md](pytorch/tensors.md) | Tensors vs NumPy, creating tensors, device placement, `requires_grad` |
| [broadcasting.md](pytorch/broadcasting.md) | Shape notation, broadcasting rules, what `dim=k` means, masked mean pooling |
| [grad_and_descent.md](pytorch/grad_and_descent.md) | The computation graph, where `.grad` lands, leaf tensors, `no_grad`, differentiability |
| [nn_module_and_multi_layer_networks.md](pytorch/nn_module_and_multi_layer_networks.md) | How `nn.Module` registers parameters, `Sequential` vs explicit `forward`, why nonlinearity |
| [linear_layer_backward.md](pytorch/linear_layer_backward.md) | Why `∂L/∂W = Xᵀ G`: where the transpose in a linear layer's backward pass comes from |
| [relu_and_dead_neurons.md](pytorch/relu_and_dead_neurons.md) | Why ReLU's zero gradient doesn't stop backprop, and when a unit really dies |
| [dataloading.md](pytorch/dataloading.md) | Mini-batches, `Dataset` vs `DataLoader`, transforms, normalization, logits |
| [inspecting_your_model_best_practices.md](pytorch/inspecting_your_model_best_practices.md) | Shape checks, hooks, runtime shape contracts, graph visualization |

### `training/`

| Page | What it covers |
|---|---|
| [training_loop.md](training/training_loop.md) | The 5-step rhythm, `train()`/`eval()`, epoch loss aggregation, functional core vs `Trainer` |
| [optimizer_and_scheduler.md](training/optimizer_and_scheduler.md) | How model, optimizer, and scheduler share tensors; built-in schedules, warmup, Noam, `ReduceLROnPlateau`, schedulers with `fit`, checkpointing |
| [adam_and_learning_rate.md](training/adam_and_learning_rate.md) | What `lr` controls, Adam line by line, AdamW, picking an `lr`, batch-size scaling |
| [observability.md](training/observability.md) | Built-in epoch telemetry, `torch.profiler`, step-level logging, async GPU timing |

### `math/`

| Page | What it covers |
|---|---|
| [matrix_calculus.md](math/matrix_calculus.md) | Jacobians, element-wise and reduction rules, the vector chain rule, a neuron's loss gradient (adapted from Parr & Howard) |

### `vision/`

| Page | What it covers |
|---|---|
| [cnn.md](vision/cnn.md) | Filters, channels, flattening, and the parameter count of the MNIST CNN |

### `transformer/`

| Page | What it covers |
|---|---|
| [attention_masks.md](transformer/attention_masks.md) | Cross-attention, encoder and decoder masks, padding masks, `masked_fill` and `-inf` |
| [norm.md](transformer/norm.md) | LayerNorm, pre-norm vs post-norm, FFN width and activation, why post-norm needs warmup |

______________________________________________________________________

## Concepts

### Tensors and shapes

| Question | Section |
|---|---|
| How is a tensor different from a NumPy array? | [tensors.md › Tensors vs NumPy](pytorch/tensors.md#tensors-vs-numpy) |
| How do I move tensors between CPU and GPU? | [tensors.md › Device Placement](pytorch/tensors.md#device-placement) |
| What does `(3,)` mean, and how is it different from `(3, 1)`? | [broadcasting.md › Q: shape (3,)](pytorch/broadcasting.md#q--shape-3-what-does-this-mean) |
| When are two shapes broadcastable? | [broadcasting.md › broadcastable](pytorch/broadcasting.md#q-what-does-broadcastable-mean) |
| What does `dim=k` mean for `sum`, `softmax`, `cat`? | [broadcasting.md › `dim=k`](pytorch/broadcasting.md#q-what-does-dimk-actually-mean) |
| How does masked mean pooling avoid NaN? | [broadcasting.md › masked mean pooling](pytorch/broadcasting.md#q-how-does-masked-mean-pooling-work) |

### Autograd and backprop

| Question | Section |
|---|---|
| What is the computation graph? | [grad_and_descent.md › The Computation Graph](pytorch/grad_and_descent.md#the-computation-graph) |
| Why does `.grad` land on the leaf? | [grad_and_descent.md › Why `.grad` Lands on the Leaf](pytorch/grad_and_descent.md#why-grad-lands-on-the-leaf-not-the-output) |
| Leaf vs non-leaf, `retain_grad`, `no_grad` | [grad_and_descent.md › Leaf vs Non-Leaf](pytorch/grad_and_descent.md#leaf-vs-non-leaf-retain_grad-and-no_grad) |
| Why do centered inputs stabilize gradients? | [grad_and_descent.md › Centered Inputs](pytorch/grad_and_descent.md#why-centered-inputs-stabilize-gradients) |
| Which losses and activations can autograd handle? | [grad_and_descent.md › Differentiability](pytorch/grad_and_descent.md#differentiability--three-cases) |
| What is a Jacobian? | [matrix_calculus.md › 4.1](math/matrix_calculus.md#41-generalization-of-the-jacobian) |
| The vector chain rule | [matrix_calculus.md › 4.5](math/matrix_calculus.md#45-the-chain-rules) |
| The gradient of a neuron's loss | [matrix_calculus.md › 6](math/matrix_calculus.md#6-the-gradient-of-the-neural-network-loss-function) |
| Where the transpose in `∂L/∂W` comes from | [linear_layer_backward.md](pytorch/linear_layer_backward.md) |
| Does ReLU's zero gradient stop backprop? | [relu_and_dead_neurons.md › Why Backprop Doesn't Stop](pytorch/relu_and_dead_neurons.md#why-backprop-doesnt-stop) |
| What is a dead neuron? | [relu_and_dead_neurons.md › When a Unit Really Dies](pytorch/relu_and_dead_neurons.md#when-a-unit-really-dies) |

### Modules and models

| Question | Section |
|---|---|
| How does `nn.Module` find my parameters? | [nn_module › `__setattr__` Interception](pytorch/nn_module_and_multi_layer_networks.md#__setattr__-interception) |
| Parameters vs buffers | [nn_module › Parameters, Buffers, and State](pytorch/nn_module_and_multi_layer_networks.md#parameters-buffers-and-state) |
| `nn.Sequential` or an explicit `forward`? | [nn_module › The Two Idioms](pytorch/nn_module_and_multi_layer_networks.md#the-two-idioms) |
| Why call `model(x)`, never `model.forward(x)`? | [nn_module › `model(x)` vs `model.forward(x)`](pytorch/nn_module_and_multi_layer_networks.md#modelx-vs-modelforwardx--never-call-forward-directly) |
| Why does a network need nonlinearity? | [nn_module › Why You Need Nonlinearity](pytorch/nn_module_and_multi_layer_networks.md#why-you-need-nonlinearity) |
| How does a convolution filter work? | [cnn.md › How One Filter Works](vision/cnn.md#how-one-filter-works) |
| How many parameters does a CNN have? | [cnn.md › Parameter Count](vision/cnn.md#parameter-count-cnnclassifier) |

### Data

| Question | Section |
|---|---|
| Why mini-batches? | [dataloading.md › Why mini-batches?](pytorch/dataloading.md#why-mini-batches) |
| `Dataset` vs `DataLoader` | [dataloading.md › Two abstractions](pytorch/dataloading.md#two-abstractions--keep-them-separate) |
| Transforms and normalization | [dataloading.md › Transforms](pytorch/dataloading.md#transforms--format-and-normalization-pipeline) |
| What are logits? | [dataloading.md › What are Logits?](pytorch/dataloading.md#what-are-logits) |
| How does a batch become a scalar loss? | [dataloading.md › Batch to Scalar Loss](pytorch/dataloading.md#how-a-batch-becomes-a-scalar-loss) |

### Training loop

| Question | Section |
|---|---|
| The five steps of every training loop | [training_loop.md › The 5-Step Rhythm](training/training_loop.md#the-5-step-rhythm) |
| Why move the model to the device before building the optimizer? | [training_loop.md › `.to(device)`](training/training_loop.md#todevice-and-the-optimizer) |
| What do `train()` and `eval()` change? | [training_loop.md › `train()` and `eval()`](training/training_loop.md#train-and-eval) |
| How do I average loss over an epoch correctly? | [training_loop.md › Aggregating Loss](training/training_loop.md#aggregating-loss-correctly-across-an-epoch) |
| Why no `Trainer` class? | [training_loop.md › Structuring the Loop](training/training_loop.md#structuring-the-loop--two-shapes) |
| How does the optimizer know which tensors to update? | [optimizer_and_scheduler.md › How the Optimizer "Knows"](training/optimizer_and_scheduler.md#how-the-optimizer-knows-which-params-to-update) |
| How do I time a training step on a GPU? | [observability.md › The Async Gotcha](training/observability.md#the-async-gotcha-that-invalidates-naive-timing) |
| How do I profile a few batches? | [observability.md › Path 1](training/observability.md#path-1--torchprofiler-scheduled-window) |

### Optimizers and learning rates

| Question | Section |
|---|---|
| My loss is `nan` / oscillating / plateauing | [adam_and_learning_rate.md › Symptoms](training/adam_and_learning_rate.md#symptoms) |
| What does Adam actually compute? | [adam_and_learning_rate.md › Adam, Line by Line](training/adam_and_learning_rate.md#adam-line-by-line) |
| Adam vs AdamW | [adam_and_learning_rate.md › Adam vs AdamW](training/adam_and_learning_rate.md#adam-vs-adamw-where-weight-decay-goes) |
| What `lr` should I start with? | [adam_and_learning_rate.md › Starting points](training/adam_and_learning_rate.md#starting-points) |
| How do I find an `lr` empirically? | [adam_and_learning_rate.md › The LR range test](training/adam_and_learning_rate.md#the-lr-range-test) |
| I changed the batch size. What happens to `lr`? | [adam_and_learning_rate.md › Batch size](training/adam_and_learning_rate.md#batch-size-moves-the-learning-rate) |
| Is my `lr` too big or too small right now? | [adam_and_learning_rate.md › Inspecting a Live Optimizer](training/adam_and_learning_rate.md#inspecting-a-live-optimizer) |

### Schedulers

| Question | Section |
|---|---|
| What does a scheduler change, and in what order do I call it? | [optimizer_and_scheduler.md › The Scheduler](training/optimizer_and_scheduler.md#the-scheduler) |
| Step per batch or per epoch? | [optimizer_and_scheduler.md › Ordering rule and cadence](training/optimizer_and_scheduler.md#ordering-rule-and-cadence) |
| `StepLR`, cosine, `OneCycleLR`: what does each do? | [optimizer_and_scheduler.md › The Built-in Schedules](training/optimizer_and_scheduler.md#the-built-in-schedules) |
| How do I chain warmup into cosine? | [optimizer_and_scheduler.md › Warmup and Decay](training/optimizer_and_scheduler.md#warmup-and-decay) |
| Why does Adam want warmup? | [adam_and_learning_rate.md › Warmup and Decay](training/adam_and_learning_rate.md#warmup-and-decay) |
| The original Transformer schedule | [optimizer_and_scheduler.md › The Noam schedule](training/optimizer_and_scheduler.md#the-noam-schedule) |
| Reduce `lr` when validation loss stalls | [optimizer_and_scheduler.md › ReduceLROnPlateau](training/optimizer_and_scheduler.md#reducelronplateau) |
| Which schedulers work with `fit`? | [optimizer_and_scheduler.md › Using a Scheduler with `fit`](training/optimizer_and_scheduler.md#using-a-scheduler-with-fit) |
| How do I resume a scheduled run? | [optimizer_and_scheduler.md › Checkpointing](training/optimizer_and_scheduler.md#checkpointing-and-reading-the-lr) |

### Debugging and inspection

| Question | Section |
|---|---|
| How do I catch shape bugs early? | [inspecting › Shape Checking](pytorch/inspecting_your_model_best_practices.md#shape-checking) |
| What are forward and backward hooks for? | [inspecting › PyTorch Hooks](pytorch/inspecting_your_model_best_practices.md#pytorch-hooks) |
| Runtime shape contracts | [inspecting › `jaxtyping` + `beartype`](pytorch/inspecting_your_model_best_practices.md#jaxtyping--beartype--runtime-shape-contracts) |
| How do I draw the model's graph? | [inspecting › Graph Visualization](pytorch/inspecting_your_model_best_practices.md#graph-visualization), [cnn.md › Visualizing the Model](vision/cnn.md#visualizing-the-model) |

### Transformers

| Question | Section |
|---|---|
| What is cross-attention? | [attention_masks.md › cross-attention](transformer/attention_masks.md#q-eli5-cross-attention-i-understand-attention-as-how-can-a-token-be-best-represented-by-other-tokens-for-a-head) |
| Which masks do the encoder and decoder use, and why? | [attention_masks.md › masks](transformer/attention_masks.md#q-explain-the-masks-used-in-the-encoder-and-decoder-why-are-they-set-up-this-way) |
| How does `padding_keep_mask` work? | [attention_masks.md › `padding_keep_mask`](transformer/attention_masks.md#q-i-cant-get-past-the-syntax-of-padding_keep_mask) |
| `~keep`, `masked_fill`, and `-inf` | [attention_masks.md › `masked_fill`](transformer/attention_masks.md#q-what-do-keep-masked_fill-and-float-inf-each-do) |
| Does `-inf` work on Mac MPS? | [attention_masks.md › MPS](transformer/attention_masks.md#q-does-float-inf-work-on-mac-mps) |
| What does LayerNorm do? | [norm.md › What Layer Normalization Does](transformer/norm.md#what-layer-normalization-does) |
| Pre-norm vs post-norm | [norm.md › Where LayerNorm Goes](transformer/norm.md#pre-norm-vs-post-norm-where-layernorm-goes) |
| Why does post-norm break as depth grows? | [norm.md › number of layers](transformer/norm.md#q-why-is-the-number-of-layers-what-breaks-post-norm) |
| How wide is the FFN, and ReLU or GELU? | [norm.md › FFN width](transformer/norm.md#q-how-wide-is-the-ffn-sublayer), [norm.md › ReLU or GELU](transformer/norm.md#q-relu-or-gelu-inside-the-ffn) |
