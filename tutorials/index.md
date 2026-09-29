# Theory & Revision

These pages explain the ideas behind the code in Dawnshard. Use a
[revision route](#revision-routes) when you want to reconnect ideas, the
[page list](#pages) when you know the topic, or the [question index](#concepts)
when one detail has gone fuzzy.

Each tutorial now starts with the central idea and uses worked explanations to
build up the precise rule. For a quick review, read its opening summary, try the
**Quick Recall** questions near the end, and open the answers after you have a
prediction. The longer derivations remain available when you need to recover
*why* a rule works.

## Revision Routes

### A batch from input to update

**Question:** How do data, a model, a loss, and an optimizer produce one weight
update? Review [tensors](pytorch/tensors.md) and
[shapes](pytorch/broadcasting.md), then trace
[modules](pytorch/nn_module_and_multi_layer_networks.md) →
[data loading](pytorch/dataloading.md) →
[the training loop](training/training_loop.md). Use
[optimizer and scheduler](training/optimizer_and_scheduler.md) to see where
the mutable state lives. The key thread is **shape → prediction → loss →
gradient → parameter change**.

### A gradient back through the model

**Question:** Why does a particular parameter receive the gradient it does?
Start with [autograd](pytorch/grad_and_descent.md), then follow the worked
[linear-layer backward](pytorch/linear_layer_backward.md) and
[ReLU](pytorch/relu_and_dead_neurons.md) explanations. Open
[matrix calculus](math/matrix_calculus.md) when you want the Jacobian rules
behind those steps. The key thread is **local derivative → chain rule →
parameter gradient**.

### Data order and training behavior

**Question:** Why can two runs, or two batches, behave differently? Review
[DataLoader randomness and workers](pytorch/randomness_and_workers.md),
[training observability](training/observability.md), and
[Adam and learning rate](training/adam_and_learning_rate.md). The key thread is
**which state changes, when it changes, and what you can measure**.

### Attention and gradient flow

**Question:** What can a token attend to, and how can gradients move through
the block? Review [attention masks](transformer/attention_masks.md) and
[pre-norm vs post-norm](transformer/norm.md). The key thread is **allowed
positions → weighted information → residual path**. The
[attention implementation notes](../app/src/training/transformer/README.md)
show where those ideas appear in the source.

### Planning a transformer run

**Question:** How long will training take, and what has to fit on each GPU?
Use [transformer compute and memory](transformer/compute_and_memory.md) to
estimate work, training time, and per-GPU state. The key thread is **tokens and
parameters → FLOPs → time; weights and activations → bytes per GPU**.

## Browse by Topic

Folders group pages by topic:

| Folder | Topic |
|---|---|
| `pytorch/` | PyTorch basics: tensors, shapes, autograd, modules, data, debugging |
| `training/` | The training loop, optimizers, schedulers, learning rates, observability |
| `math/` | The calculus under backprop |
| `vision/` | Convolutional networks |
| `transformer/` | Attention, masks, normalization, compute and memory budgets |

If the vocabulary is new, begin with the short
[PyTorch overview](pytorch/overview.md), then choose the first route above that
answers your question.

______________________________________________________________________

## Pages

### `pytorch/`

| Page | Use it to revisit |
|---|---|
| [overview.md](pytorch/overview.md) | Where the five parts of a PyTorch model fit together |
| [tensors.md](pytorch/tensors.md) | What device, storage, and gradient tracking add to an array |
| [broadcasting.md](pytorch/broadcasting.md) | Why shapes expand and where broadcast gradients go |
| [grad_and_descent.md](pytorch/grad_and_descent.md) | How autograd records operations and fills parameter gradients |
| [nn_module_and_multi_layer_networks.md](pytorch/nn_module_and_multi_layer_networks.md) | How modules own parameters and why layers need nonlinearity |
| [linear_layer_backward.md](pytorch/linear_layer_backward.md) | Where the transpose in `∂L/∂W = Xᵀ G` comes from |
| [relu_and_dead_neurons.md](pytorch/relu_and_dead_neurons.md) | When ReLU blocks a gradient and when a unit is truly dead |
| [dataloading.md](pytorch/dataloading.md) | How samples become batches, logits, and a scalar loss |
| [randomness_and_workers.md](pytorch/randomness_and_workers.md) | Seed vs state, what a generator stores, where shuffling occurs, and why workers need separate random streams |
| [inspecting_your_model_best_practices.md](pytorch/inspecting_your_model_best_practices.md) | How shapes, hooks, and graph views reveal model mistakes |

### `training/`

| Page | Use it to revisit |
|---|---|
| [training_loop.md](training/training_loop.md) | How one batch updates weights and how epoch results are combined |
| [optimizer_and_scheduler.md](training/optimizer_and_scheduler.md) | Which state the model, optimizer, and scheduler share or change |
| [adam_and_learning_rate.md](training/adam_and_learning_rate.md) | Why Adam adapts steps and what the learning rate controls |
| [observability.md](training/observability.md) | Which measurements explain slow or unstable runs |

### `math/`

| Page | Use it to revisit |
|---|---|
| [matrix_calculus.md](math/matrix_calculus.md) | How Jacobian shapes and the chain rule explain a neuron's gradient |

### `vision/`

| Page | Use it to revisit |
|---|---|
| [cnn.md](vision/cnn.md) | How filters change channel shapes and parameter counts |

### `transformer/`

| Page | Use it to revisit |
|---|---|
| [attention_masks.md](transformer/attention_masks.md) | Which attention pairs are allowed and how a mask enforces that |
| [norm.md](transformer/norm.md) | How normalization placement changes forward and gradient paths |
| [compute_and_memory.md](transformer/compute_and_memory.md) | How parameters, tokens, precision, activations, and parallelism determine run cost |

______________________________________________________________________

## Concepts

### Tensors and shapes

| Question | Section |
|---|---|
| How is a tensor different from a NumPy array? | [tensors.md › Tensors vs NumPy](pytorch/tensors.md#tensors-vs-numpy) |
| How do I move tensors between CPU and GPU? | [tensors.md › Device Placement](pytorch/tensors.md#device-placement) |
| What does `(3,)` mean, and how is it different from `(3, 1)`? | [broadcasting.md › Q: shape (3,)](pytorch/broadcasting.md#q--shape-3-what-does-this-mean) |
| When are two shapes broadcastable? | [broadcasting.md › broadcastable](pytorch/broadcasting.md#q-what-does-broadcastable-mean) |
| `expand` or `repeat`? | [broadcasting.md › `expand` vs `repeat`](pytorch/broadcasting.md#q-what-is-the-difference-between-expand-and-repeat) |
| Does `@` broadcast like `+` does? | [broadcasting.md › matmul](pytorch/broadcasting.md#q-does-matmul-broadcast-the-same-way) |
| Why is my bias gradient the bias's shape, not the batch's? | [broadcasting.md › gradients](pytorch/broadcasting.md#q-what-happens-to-gradients-when-a-tensor-is-broadcast) |
| What does `dim=k` mean for `sum`, `softmax`, `cat`? | [broadcasting.md › `dim=k`](pytorch/broadcasting.md#q-what-does-dimk-actually-mean) |
| How does masked mean pooling avoid NaN? | [broadcasting.md › masked mean pooling](pytorch/broadcasting.md#q-how-does-masked-mean-pooling-work) |

### Autograd and backprop

| Question | Section |
|---|---|
| What is the computation graph? | [grad_and_descent.md › The Computation Graph](pytorch/grad_and_descent.md#the-computation-graph) |
| Why does `.grad` land on the leaf? | [grad_and_descent.md › Why `.grad` Lands on the Leaf](pytorch/grad_and_descent.md#why-grad-lands-on-the-leaf-not-the-output) |
| Leaf vs non-leaf, `retain_grad`, `no_grad` | [grad_and_descent.md › Leaf vs Non-Leaf](pytorch/grad_and_descent.md#leaf-vs-non-leaf-retain_grad-and-no_grad) |
| Where does the gradient go when a tensor was broadcast? | [broadcasting.md › gradients](pytorch/broadcasting.md#q-what-happens-to-gradients-when-a-tensor-is-broadcast) |
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
| Does a random draw change the seed? Seed vs state, and the global generator | [randomness_and_workers.md › seed vs state](pytorch/randomness_and_workers.md#q-after-torchmanual_seeda-and-one-draw-is-the-seed-now-a--1) |
| Does a generator store its random numbers? How does a draw update the state? | [randomness_and_workers.md › state and formula](pytorch/randomness_and_workers.md#q-does-a-generator-store-an-array-of-random-numbers) |
| How do I keep one component's randomness from shifting another's? | [randomness_and_workers.md › `torch.Generator`](pytorch/randomness_and_workers.md#q-what-is-a-torchgenerator-and-why-use-one) |
| Why does `torch.Generator()` in a loop return the same value every time? | [randomness_and_workers.md › unseeded generator](pytorch/randomness_and_workers.md#an-unseeded-generator-always-starts-from-the-same-seed) |
| Does `generator=g` advance the global generator too? | [randomness_and_workers.md › one generator per draw](pytorch/randomness_and_workers.md#a-draw-advances-only-the-generator-it-reads) |
| Map-style vs iterable-style datasets | [randomness_and_workers.md › map-style](pytorch/randomness_and_workers.md#q-the-map-style-dataset-is-then-passed-to-the-dataloader-to-create-batches-what-does-that-mean) |
| Who shuffles, and why do workers get a seed? | [randomness_and_workers.md › worker seeds](pytorch/randomness_and_workers.md#q-why-does-the-dataloader-pass-an-rng-seed-to-workers-it-could-just-shuffle-the-indices-and-send-them) |
| How does a worker's seed trace back to `torch.manual_seed`? | [randomness_and_workers.md › seed chain](pytorch/randomness_and_workers.md#q-does-a-workers-randomness-depend-on-the-global-generator-or-on-the-base-seed) |
| Why do my random masks repeat every epoch? | [randomness_and_workers.md › `seed + index`](pytorch/randomness_and_workers.md#q-a-collate_fn-seeds-each-sample-with-seed--index-do-its-masks-repeat) |
| Why does validating change my training run? `fork_rng` | [randomness_and_workers.md › `fork_rng`](pytorch/randomness_and_workers.md#q-what-does-torchrandomfork_rng-do) |
| Are workers threads or processes? `fork` or `spawn`? | [randomness_and_workers.md › processes](pytorch/randomness_and_workers.md#q-with-num_workers--0-is-each-worker-a-thread-or-a-process) |
| Can a forked worker use CUDA? | [randomness_and_workers.md › CUDA and `fork`](pytorch/randomness_and_workers.md#q-isnt-forking-a-process-that-uses-cuda-forbidden) |

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
| Why is `normalized_shape` set to `d_model`? | [norm.md › `normalized_shape`](transformer/norm.md#which-axes-normalized_shape) |
| Why is it called *layer* norm if it normalizes per token? | [norm.md › Why it's called layer norm](transformer/norm.md#why-its-called-layer-norm) |
| Pre-norm vs post-norm | [norm.md › Where LayerNorm Goes](transformer/norm.md#pre-norm-vs-post-norm-where-layernorm-goes) |
| Why does post-norm break as depth grows? | [norm.md › number of layers](transformer/norm.md#q-why-is-the-number-of-layers-what-breaks-post-norm) |
| How wide is the FFN, and ReLU or GELU? | [norm.md › FFN width](transformer/norm.md#q-how-wide-is-the-ffn-sublayer), [norm.md › ReLU or GELU](transformer/norm.md#q-relu-or-gelu-inside-the-ffn) |
| How do I estimate transformer training FLOPs and time? | [compute_and_memory.md › Count the Training Work](transformer/compute_and_memory.md#count-the-training-work) |
| How should parameters and training tokens share a compute budget? | [compute_and_memory.md › Parameters and Tokens](transformer/compute_and_memory.md#parameters-and-tokens-are-different-choices) |
| How much memory does inference need beyond model weights? | [compute_and_memory.md › Inference Memory](transformer/compute_and_memory.md#inference-memory-weights-and-the-kv-cache) |
| What makes up a mixed-precision Adam training budget? | [compute_and_memory.md › A Worked Training Budget](transformer/compute_and_memory.md#a-worked-training-budget) |
| How does activation checkpointing trade compute for memory? | [compute_and_memory.md › Activations](transformer/compute_and_memory.md#activations-grow-with-the-microbatch) |
| What do ZeRO-1, ZeRO-2, and ZeRO-3 shard? | [compute_and_memory.md › What ZeRO Shards](transformer/compute_and_memory.md#what-zero-shards) |
| How many data-parallel replicas are in a 3D parallel run? | [compute_and_memory.md › Parallelism](transformer/compute_and_memory.md#how-data-tensor-and-pipeline-parallelism-combine) |
