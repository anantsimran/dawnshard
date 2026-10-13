# Theory & Revision

These pages explain the ideas behind the code in Dawnshard. Use a
[revision route](#revision-routes) when you want to reconnect ideas, the
[page list](#pages) when you know the topic, or the [question index](#concepts)
when one detail has gone fuzzy.

There is one file per topic, and each tutorial is a `##` section inside it. So
every answer for a topic is on one page: search it with one `ctrl-F`, and read a
topic straight through when you want the whole arc rather than one answer.

Each tutorial starts with the central idea and uses worked explanations to
build up the precise rule. For a quick review, read its opening summary, try the
recall questions at the end of that section, and open the answers after you have a
prediction. The longer derivations remain available when you need to recover
*why* a rule works.

## Revision Routes

### A batch from input to update

**Question:** How do data, a model, a loss, and an optimizer produce one weight
update? Review [tensors](pytorch.md#tensors) and
[shapes](pytorch.md#shapes-and-broadcasting--qa), then trace
[modules](pytorch.md#nnmodule-and-multi-layer-networks) →
[data loading](pytorch.md#dataloader--batching-and-the-dataset-abstraction) →
[the training loop](training.md#the-training-loop). Use
[optimizer and scheduler](training.md#optimizer--scheduler--the-decoupling) to see where
the mutable state lives. The key thread is **shape → prediction → loss →
gradient → parameter change**.

### A gradient back through the model

**Question:** Why does a particular parameter receive the gradient it does?
Start with [autograd](pytorch.md#gradients-and-gradient-descent), then follow the worked
[linear-layer backward](pytorch.md#where-the-transpose-comes-from) and
[ReLU](pytorch.md#relu-and-dead-neurons) explanations. Open
[matrix calculus](math.md#the-matrix-calculus-you-need-for-deep-learning) when you want the Jacobian rules
behind those steps. The key thread is **local derivative → chain rule →
parameter gradient**.

### Data order and training behavior

**Question:** Why can two runs, or two batches, behave differently? Review
[DataLoader randomness and workers](pytorch.md#random-streams-and-dataloader-workers--qa),
[training observability](training.md#observability-in-the-training-loop), and
[Adam and learning rate](training.md#adam-and-the-learning-rate). The key thread is
**which state changes, when it changes, and what you can measure**.

### Attention and gradient flow

**Question:** What can a token attend to, and how can gradients move through
the block? Review [attention masks](transformer.md#attention-masks-and-cross-attention--qa) and
[pre-norm vs post-norm](transformer.md#pre-norm-vs-post-norm). The key thread is **allowed
positions → weighted information → residual path**. The
[attention implementation notes](../app/src/training/transformer/README.md)
show where those ideas appear in the source.

### Planning a transformer run

**Question:** How long will training take, and what has to fit on each GPU?
Use [transformer compute and memory](transformer.md#transformer-compute-and-memory) to
estimate work, training time, and per-GPU state. The key thread is **tokens and
parameters → FLOPs → time; weights and activations → bytes per GPU**.

## Browse by Topic

One file per topic. Each tutorial is a `##` section inside it, so a topic reads
start to finish, and one page holds every answer for that topic:

| Page | Topic |
|---|---|
| [pytorch.md](pytorch.md) | PyTorch basics: tensors, shapes, autograd, modules, data, debugging |
| [training.md](training.md) | The training loop, optimizers, schedulers, learning rates, observability |
| [transformer.md](transformer.md) | Attention, masks, normalization, compute and memory budgets |
| [math.md](math.md) | The calculus under backprop |
| [vision.md](vision.md) | Convolutional networks |

If the vocabulary is new, begin with the short
[PyTorch overview](pytorch.md#pytorch--overview), then choose the first route above that
answers your question.

______________________________________________________________________

## Pages

Each row is a `##` section of its topic file.

### [pytorch.md](pytorch.md)

| Section | Use it to revisit |
|---|---|
| [PyTorch — Overview](pytorch.md#pytorch--overview) | Where the five parts of a PyTorch model fit together |
| [Tensors](pytorch.md#tensors) | What device, storage, and gradient tracking add to an array |
| [Shapes and Broadcasting](pytorch.md#shapes-and-broadcasting--qa) | Why shapes expand, how `cat` and `stack` join tensors, how one `nn.Linear` serves every token, and where broadcast gradients go |
| [Gradients and Gradient Descent](pytorch.md#gradients-and-gradient-descent) | How autograd records operations and fills parameter gradients |
| [`nn.Module` and Multi-Layer Networks](pytorch.md#nnmodule-and-multi-layer-networks) | How modules own parameters and why layers need nonlinearity |
| [Where the Transpose Comes From](pytorch.md#where-the-transpose-comes-from) | Where the transpose in `∂L/∂W = Xᵀ G` comes from |
| [ReLU and Dead Neurons](pytorch.md#relu-and-dead-neurons) | When ReLU blocks a gradient and when a unit is truly dead |
| [DataLoader](pytorch.md#dataloader--batching-and-the-dataset-abstraction) | How samples become batches, logits, and a scalar loss |
| [Random Streams and DataLoader Workers](pytorch.md#random-streams-and-dataloader-workers--qa) | Seed vs state, what a generator stores, where shuffling occurs, and why workers need separate random streams |
| [Inspecting Your Model](pytorch.md#inspecting-your-model--best-practices) | How shapes, hooks, and graph views reveal model mistakes |

### [training.md](training.md)

| Section | Use it to revisit |
|---|---|
| [The Training Loop](training.md#the-training-loop) | How one batch updates weights and how epoch results are combined |
| [Optimizer & Scheduler](training.md#optimizer--scheduler--the-decoupling) | Which state the model, optimizer, and scheduler share or change |
| [Adam and the Learning Rate](training.md#adam-and-the-learning-rate) | Why Adam adapts steps and what the learning rate controls |
| [Observability](training.md#observability-in-the-training-loop) | How to time setup work and investigate slow or unstable runs |

### [transformer.md](transformer.md)

| Section | Use it to revisit |
|---|---|
| [Attention Masks and Cross-Attention](transformer.md#attention-masks-and-cross-attention--qa) | Which attention pairs are allowed, how a mask enforces that, and what flows through a decoder block |
| [Pre-Norm vs. Post-Norm](transformer.md#pre-norm-vs-post-norm) | How normalization placement changes forward and gradient paths |
| [Transformer Compute and Memory](transformer.md#transformer-compute-and-memory) | How parameters, tokens, precision, activations, and parallelism determine run cost |

### [math.md](math.md)

| Section | Use it to revisit |
|---|---|
| [The Matrix Calculus You Need for Deep Learning](math.md#the-matrix-calculus-you-need-for-deep-learning) | How Jacobian shapes and the chain rule explain a neuron's gradient |

### [vision.md](vision.md)

| Section | Use it to revisit |
|---|---|
| [Convolutional Neural Networks](vision.md#convolutional-neural-networks) | How filters change channel shapes and parameter counts |

______________________________________________________________________

## Concepts

### Tensors and shapes

| Question | Section |
|---|---|
| How is a tensor different from a NumPy array? | [Tensors › Tensors vs NumPy](pytorch.md#tensors-vs-numpy) |
| How do I move tensors between CPU and GPU? | [Tensors › Device Placement](pytorch.md#device-placement) |
| What does `(3,)` mean, and how is it different from `(3, 1)`? | [Shapes and Broadcasting › Q: shape (3,)](pytorch.md#q--shape-3-what-does-this-mean) |
| When are two shapes broadcastable? | [Shapes and Broadcasting › broadcastable](pytorch.md#q-what-does-broadcastable-mean) |
| `expand` or `repeat`? | [Shapes and Broadcasting › `expand` vs `repeat`](pytorch.md#q-what-is-the-difference-between-expand-and-repeat) |
| Does `@` broadcast like `+` does? | [Shapes and Broadcasting › matmul](pytorch.md#q-does-matmul-broadcast-the-same-way) |
| Does every token get the same `nn.Linear` weight? Does a GPU run the batch one token at a time? | [Shapes and Broadcasting › shared weights](pytorch.md#q-how-does-nnlinear-apply-one-weight-to-every-token) |
| Why is my bias gradient the bias's shape, not the batch's? | [Shapes and Broadcasting › gradients](pytorch.md#q-what-happens-to-gradients-when-a-tensor-is-broadcast) |
| What does `dim=k` mean for `sum`, `softmax`, `cat`? | [Shapes and Broadcasting › `dim=k`](pytorch.md#q-what-does-dimk-actually-mean) |
| Is `torch.tensor([1, 2])` a row or a column? | [Shapes and Broadcasting › orientation](pytorch.md#q-is-torchtensor1-2-a-row-or-a-column) |
| `cat` or `stack`? What does each `dim` give me? | [Shapes and Broadcasting › `cat` or `stack`](pytorch.md#q-cat-or-stack) |
| How does masked mean pooling avoid NaN? | [Shapes and Broadcasting › masked mean pooling](pytorch.md#q-how-does-masked-mean-pooling-work) |

### Autograd and backprop

| Question | Section |
|---|---|
| What is the computation graph? | [Gradients and Gradient Descent › The Computation Graph](pytorch.md#the-computation-graph) |
| Why does `.grad` land on the leaf? | [Gradients and Gradient Descent › Why `.grad` Lands on the Leaf](pytorch.md#why-grad-lands-on-the-leaf-not-the-output) |
| Leaf vs non-leaf, `retain_grad`, `no_grad` | [Gradients and Gradient Descent › Leaf vs Non-Leaf](pytorch.md#leaf-vs-non-leaf-retain_grad-and-no_grad) |
| Where does the gradient go when a tensor was broadcast? | [Shapes and Broadcasting › gradients](pytorch.md#q-what-happens-to-gradients-when-a-tensor-is-broadcast) |
| Why do centered inputs stabilize gradients? | [Gradients and Gradient Descent › Centered Inputs](pytorch.md#why-centered-inputs-stabilize-gradients) |
| Which losses and activations can autograd handle? | [Gradients and Gradient Descent › Differentiability](pytorch.md#differentiability--three-cases) |
| What is a Jacobian? | [Matrix Calculus › 4.1](math.md#41-generalization-of-the-jacobian) |
| The vector chain rule | [Matrix Calculus › 4.5](math.md#45-the-chain-rules) |
| The gradient of a neuron's loss | [Matrix Calculus › 6](math.md#6-the-gradient-of-the-neural-network-loss-function) |
| Where the transpose in `∂L/∂W` comes from | [Where the Transpose Comes From](pytorch.md#where-the-transpose-comes-from) |
| Does ReLU's zero gradient stop backprop? | [ReLU and Dead Neurons › Why Backprop Doesn't Stop](pytorch.md#why-backprop-doesnt-stop) |
| What is a dead neuron? | [ReLU and Dead Neurons › When a Unit Really Dies](pytorch.md#when-a-unit-really-dies) |

### Modules and models

| Question | Section |
|---|---|
| How does `nn.Module` find my parameters? | [nn_module › `__setattr__` Interception](pytorch.md#__setattr__-interception) |
| Parameters vs buffers | [nn_module › Parameters, Buffers, and State](pytorch.md#parameters-buffers-and-state) |
| `nn.Sequential` or an explicit `forward`? | [nn_module › The Two Idioms](pytorch.md#the-two-idioms) |
| Why call `model(x)`, never `model.forward(x)`? | [nn_module › `model(x)` vs `model.forward(x)`](pytorch.md#modelx-vs-modelforwardx--never-call-forward-directly) |
| Why does a network need nonlinearity? | [nn_module › Why You Need Nonlinearity](pytorch.md#why-you-need-nonlinearity) |
| How does a convolution filter work? | [Convolutional Neural Networks › How One Filter Works](vision.md#how-one-filter-works) |
| How many parameters does a CNN have? | [Convolutional Neural Networks › Parameter Count](vision.md#parameter-count-convolutionalmnistclassifier) |

### Data

| Question | Section |
|---|---|
| Why mini-batches? | [DataLoader › Why mini-batches?](pytorch.md#why-mini-batches) |
| `Dataset` vs `DataLoader` | [DataLoader › Two abstractions](pytorch.md#two-abstractions--keep-them-separate) |
| Transforms and normalization | [DataLoader › Transforms](pytorch.md#transforms--format-and-normalization-pipeline) |
| What are logits? | [DataLoader › What are Logits?](pytorch.md#what-are-logits) |
| How does a batch become a scalar loss? | [DataLoader › Batch to Scalar Loss](pytorch.md#how-a-batch-becomes-a-scalar-loss) |
| How does `(B, L, vocab)` output become one loss? What does `ignore_index` divide by? | [DataLoader › Sequence outputs](pytorch.md#sequence-outputs-one-loss-per-token) |
| Does a random draw change the seed? Seed vs state, and the global generator | [Random Streams and DataLoader Workers › seed vs state](pytorch.md#q-after-torchmanual_seeda-and-one-draw-is-the-seed-now-a--1) |
| Does a generator store its random numbers? How does a draw update the state? | [Random Streams and DataLoader Workers › state and formula](pytorch.md#q-does-a-generator-store-an-array-of-random-numbers) |
| How do I keep one component's randomness from shifting another's? | [Random Streams and DataLoader Workers › `torch.Generator`](pytorch.md#q-what-is-a-torchgenerator-and-why-use-one) |
| Why does `torch.Generator()` in a loop return the same value every time? | [Random Streams and DataLoader Workers › unseeded generator](pytorch.md#an-unseeded-generator-always-starts-from-the-same-seed) |
| Does `generator=g` advance the global generator too? | [Random Streams and DataLoader Workers › one generator per draw](pytorch.md#a-draw-advances-only-the-generator-it-reads) |
| Map-style vs iterable-style datasets | [Random Streams and DataLoader Workers › map-style](pytorch.md#q-the-map-style-dataset-is-then-passed-to-the-dataloader-to-create-batches-what-does-that-mean) |
| Who shuffles, and why do workers get a seed? | [Random Streams and DataLoader Workers › worker seeds](pytorch.md#q-why-does-the-dataloader-pass-an-rng-seed-to-workers-it-could-just-shuffle-the-indices-and-send-them) |
| How does a worker's seed trace back to `torch.manual_seed`? | [Random Streams and DataLoader Workers › seed chain](pytorch.md#q-does-a-workers-randomness-depend-on-the-global-generator-or-on-the-base-seed) |
| Why do my random masks repeat every epoch? | [Random Streams and DataLoader Workers › `seed + index`](pytorch.md#q-a-collate_fn-seeds-each-sample-with-seed--index-do-its-masks-repeat) |
| Why does validating change my training run? `fork_rng` | [Random Streams and DataLoader Workers › `fork_rng`](pytorch.md#q-what-does-torchrandomfork_rng-do) |
| Are workers threads or processes? `fork` or `spawn`? | [Random Streams and DataLoader Workers › processes](pytorch.md#q-with-num_workers--0-is-each-worker-a-thread-or-a-process) |
| Can a forked worker use CUDA? | [Random Streams and DataLoader Workers › CUDA and `fork`](pytorch.md#q-isnt-forking-a-process-that-uses-cuda-forbidden) |

### Training loop

| Question | Section |
|---|---|
| The five steps of every training loop | [The Training Loop › The 5-Step Rhythm](training.md#the-5-step-rhythm) |
| Why move the model to the device before building the optimizer? | [The Training Loop › `.to(device)`](training.md#todevice-and-the-optimizer) |
| What do `train()` and `eval()` change? | [The Training Loop › `train()` and `eval()`](training.md#train-and-eval) |
| How do I average loss over an epoch correctly? | [The Training Loop › Aggregating Loss](training.md#aggregating-loss-correctly-across-an-epoch) |
| Why no `Trainer` class? | [The Training Loop › Structuring the Loop](training.md#structuring-the-loop--two-shapes) |
| How does the optimizer know which tensors to update? | [Optimizer & Scheduler › How the Optimizer "Knows"](training.md#how-the-optimizer-knows-which-params-to-update) |
| How do I time a training step on a GPU? | [Observability › The Async Gotcha](training.md#the-async-gotcha-that-invalidates-naive-timing) |
| How do I log the time spent preparing data? | [Observability › Timing setup work](training.md#timing-setup-work-with-log_elapsed) |
| How do I profile a few batches? | [Observability › Path 1](training.md#path-1--torchprofiler-scheduled-window) |

### Optimizers and learning rates

| Question | Section |
|---|---|
| My loss is `nan` / oscillating / plateauing | [Adam and the Learning Rate › Symptoms](training.md#symptoms) |
| What does Adam actually compute? | [Adam and the Learning Rate › Adam, Line by Line](training.md#adam-line-by-line) |
| Adam vs AdamW | [Adam and the Learning Rate › Adam vs AdamW](training.md#adam-vs-adamw-where-weight-decay-goes) |
| What `lr` should I start with? | [Adam and the Learning Rate › Starting points](training.md#starting-points) |
| How do I find an `lr` empirically? | [Adam and the Learning Rate › The LR range test](training.md#the-lr-range-test) |
| I changed the batch size. What happens to `lr`? | [Adam and the Learning Rate › Batch size](training.md#batch-size-moves-the-learning-rate) |
| Is my `lr` too big or too small right now? | [Adam and the Learning Rate › Inspecting a Live Optimizer](training.md#inspecting-a-live-optimizer) |

### Schedulers

| Question | Section |
|---|---|
| What does a scheduler change, and in what order do I call it? | [Optimizer & Scheduler › The Scheduler](training.md#the-scheduler) |
| Step per batch or per epoch? | [Optimizer & Scheduler › Ordering rule and cadence](training.md#ordering-rule-and-cadence) |
| `StepLR`, cosine, `OneCycleLR`: what does each do? | [Optimizer & Scheduler › The Built-in Schedules](training.md#the-built-in-schedules) |
| How do I chain warmup into cosine? | [Optimizer & Scheduler › Warmup and Decay](training.md#warmup-and-decay-how-to-build-them) |
| Why does Adam want warmup? | [Adam and the Learning Rate › Warmup and Decay](training.md#warmup-and-decay-why-adam-wants-them) |
| The original Transformer schedule | [Optimizer & Scheduler › The Noam schedule](training.md#the-noam-schedule) |
| Reduce `lr` when validation loss stalls | [Optimizer & Scheduler › ReduceLROnPlateau](training.md#reducelronplateau) |
| Which schedulers work with `fit`? | [Optimizer & Scheduler › Using a Scheduler with `fit`](training.md#using-a-scheduler-with-fit) |
| How do I resume a scheduled run? | [Optimizer & Scheduler › Checkpointing](training.md#checkpointing-and-reading-the-lr) |

### Debugging and inspection

| Question | Section |
|---|---|
| How do I catch shape bugs early? | [inspecting › Shape Checking](pytorch.md#shape-checking) |
| What are forward and backward hooks for? | [inspecting › PyTorch Hooks](pytorch.md#pytorch-hooks) |
| Runtime shape contracts | [inspecting › `jaxtyping` + `beartype`](pytorch.md#jaxtyping--beartype--runtime-shape-contracts) |
| How do I draw the model's graph? | [inspecting › Graph Visualization](pytorch.md#graph-visualization), [Convolutional Neural Networks › Visualizing the Model](vision.md#visualizing-the-model) |

### Transformers

| Question | Section |
|---|---|
| What is cross-attention? | [Attention Masks › cross-attention](transformer.md#q-eli5-cross-attention-i-understand-attention-as-how-can-a-token-be-best-represented-by-other-tokens-for-a-head) |
| What flows through a decoder block, and what do its self- and cross-attention Q, K, V represent? | [Attention Masks › decoder block](transformer.md#q-what-flows-through-a-decoder-block-and-what-does-each-part-represent) |
| Why do cross-attention's values come from the encoder, not the decoder? | [Attention Masks › cross-attention values](transformer.md#q-why-must-cross-attentions-values-come-from-the-source) |
| Which masks do the encoder and decoder use, and why? | [Attention Masks › masks](transformer.md#q-explain-the-masks-used-in-the-encoder-and-decoder-why-are-they-set-up-this-way) |
| What does a causal mask look like, and what does each row predict? | [Attention Masks › causal mask](transformer.md#q-draw-a-causal-mask-does-the-position-predicting-t1-see-only-bos) |
| A position can see its own token. Why can't it copy the answer? | [Attention Masks › copying](transformer.md#q-row-le-can-see-itself-can-it-copy-the-answer) |
| How does `padding_keep_mask` work? | [Attention Masks › `padding_keep_mask`](transformer.md#q-i-cant-get-past-the-syntax-of-padding_keep_mask) |
| `~keep`, `masked_fill`, and `-inf` | [Attention Masks › `masked_fill`](transformer.md#q-what-do-keep-masked_fill-and-float-inf-each-do) |
| Does `-inf` work on Mac MPS? | [Attention Masks › MPS](transformer.md#q-does-float-inf-work-on-mac-mps) |
| What does LayerNorm do? | [Pre-Norm vs. Post-Norm › What Layer Normalization Does](transformer.md#what-layer-normalization-does) |
| Why is `normalized_shape` set to `d_model`? | [Pre-Norm vs. Post-Norm › `normalized_shape`](transformer.md#which-axes-normalized_shape) |
| Why is it called *layer* norm if it normalizes per token? | [Pre-Norm vs. Post-Norm › Why it's called layer norm](transformer.md#why-its-called-layer-norm) |
| Pre-norm vs post-norm | [Pre-Norm vs. Post-Norm › Where LayerNorm Goes](transformer.md#pre-norm-vs-post-norm-where-layernorm-goes) |
| Why does post-norm break as depth grows? | [Pre-Norm vs. Post-Norm › number of layers](transformer.md#q-why-is-the-number-of-layers-what-breaks-post-norm) |
| How wide is the FFN, and ReLU or GELU? | [Pre-Norm vs. Post-Norm › FFN width](transformer.md#q-how-wide-is-the-ffn-sublayer), [Pre-Norm vs. Post-Norm › ReLU or GELU](transformer.md#q-relu-or-gelu-inside-the-ffn) |
| How do I estimate transformer training FLOPs and time? | [Transformer Compute and Memory › Count the Training Work](transformer.md#count-the-training-work) |
| How should parameters and training tokens share a compute budget? | [Transformer Compute and Memory › Parameters and Tokens](transformer.md#parameters-and-tokens-are-different-choices) |
| How much memory does inference need beyond model weights? | [Transformer Compute and Memory › Inference Memory](transformer.md#inference-memory-weights-and-the-kv-cache) |
| What makes up a mixed-precision Adam training budget? | [Transformer Compute and Memory › A Worked Training Budget](transformer.md#a-worked-training-budget) |
| How does activation checkpointing trade compute for memory? | [Transformer Compute and Memory › Activations](transformer.md#activations-grow-with-the-microbatch) |
| What do ZeRO-1, ZeRO-2, and ZeRO-3 shard? | [Transformer Compute and Memory › What ZeRO Shards](transformer.md#what-zero-shards) |
| How many data-parallel replicas are in a 3D parallel run? | [Transformer Compute and Memory › Parallelism](transformer.md#how-data-tensor-and-pipeline-parallelism-combine) |
