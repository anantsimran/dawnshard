# Graph Report - dawnshard (2026-09-29)

## Corpus Check

- 83 files · ~95,446 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: (none) 6, .css 1, .lock 1)

## Summary

- 1001 nodes · 1857 edges · 54 communities (44 shown, 10 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 317 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness

- Built from commit: `d2b56dee`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)

- test_priority_queue.py
- index.md
- SinusoidalEmbedding
- attention
- compare_runs.py
- test_layout.py
- ag_news.py
- MaxHeap
- Getting started
- bpe.py
- mnist.py
- plot_attention_heads
- TrainState
- test_bpe.py
- multi30ken_predictor.py
- Dataload
- padding_keep_mask
- hooks.py
- test_mask.py
- .forward
- test_ag_news_classifier.py
- Shapes and Broadcasting — Q&A
- multi30ken.py
- HasAttentionMap
- .forward
- The Matrix Calculus You Need for Deep Learning
- fit
- .create
- test_multi30ken.py
- test_metrics.py
- Transformer Compute and Memory
- AGNewsClassifier
- Random Streams and DataLoader Workers — Q&A
- train
- Concepts
- ClassificationMetrics
- train_config
- inspect_mnist_dataset
- pre-commit
- DataLoader — Batching and the Dataset Abstraction
- Sunrise favicon
- dawnshard
- Training run metrics plot
- get_device
- AGENTS.md
- MathJax page override
- \_encode_rows
- .forward
- bpe_seq_len.py
- save_attention_maps

## God Nodes (most connected - your core abstractions)

1. `fit()` - 47 edges
1. `TrainState` - 35 edges
1. `EpochSpec` - 30 edges
1. `Metrics` - 26 edges
1. `profiled_fit()` - 22 edges
1. `attention()` - 22 edges
1. `Phrase` - 20 edges
1. `eval_step()` - 20 edges
1. `run_epoch()` - 20 edges
1. `make_state_and_config()` - 19 edges

## Surprising Connections (you probably didn't know these)

- `Training loop` --references--> `train()` [INFERRED]
  tutorials/index.md → app/src/training/common/bpe.py
- `Tutorials site` --references--> `EpochSpec` [INFERRED]
  CLAUDE.md → app/src/training/train/model.py
- `Probes: a function of the training state` --references--> `EpochSpec` [INFERRED]
  README.md → app/src/training/train/model.py
- `Schedulers` --references--> `fit()` [INFERRED]
  tutorials/index.md → app/src/training/train/train_loop.py
- `Transformers` --references--> `padding_keep_mask()` [INFERRED]
  tutorials/index.md → app/src/training/transformer/mask.py

## Import Cycles

- None detected.

## Hyperedges (group relationships)

- **Training components communicate through shared state** — tutorials_pytorch_nn_module_and_multi_layer_networks_module_parameter_registration, tutorials_training_optimizer_and_scheduler_shared_model_optimizer_parameters, tutorials_training_optimizer_and_scheduler_scheduler_edits_optimizer_learning_rate, tutorials_training_training_loop_five_step_training_loop [INFERRED 0.95]

## Communities (54 total, 10 thin omitted)

### Community 0 - "test_priority_queue.py"

Cohesion: 0.23
Nodes (15): \_heap(), test_add_item_rejects_duplicate(), test_add_item_then_pop_highest_priority(), test_constructor_rejects_duplicate_items(), test_get_priority(), test_heap_invariant_holds_after_mixed_operations(), test_len_and_contains(), test_pop_from_empty_heap_raises() (+7 more)

### Community 1 - "index.md"

Cohesion: 0.05
Nodes (58): Common, Autograd computation graph, Differentiability and subgradients, Grad And Descent, Leaf gradient accumulation, Autograd graph visualization, Inspecting Your Model Best Practices, Model shape contracts (+50 more)

### Community 2 - "SinusoidalEmbedding"

Cohesion: 0.06
Nodes (40): Build the embedding, encoder stack, and classifier head. Args: vocab_size:…, Build the embedding, position table, encoder stack, final norm, and head. Args:…, MultiHeadAttentionLayer, Multi-head self-attention for an encoder, with an optional padding mask.…, Raises ValueError if `d_model` is not divisible by `h`. Args: qk_norm: RMSNorm…, Add fixed sinusoidal position encodings to token embeddings. Attention is…, Build the block's two sublayers and their norms. Args: qk_norm: Forwarded to…, Precompute the (seq_length, d_model) position table. Args: seq_length: Longest… (+32 more)

### Community 3 - "attention"

Cohesion: 0.17
Nodes (13): attention(), B, Bool, Float, H, jaxtyped, L, Tensor (+5 more)

### Community 4 - "compare_runs.py"

Cohesion: 0.08
Nodes (32): load_history(), main(), Figure, Path, Compare per-epoch metrics from two training runs on the same graph., Write `fig` as a base64-embedded PNG in an HTML file and open it in the browser., Load per-epoch records, accepting either a bare JSON list or \`{"history":…, CLI entry point: plot shared metrics from two history files and open the… (+24 more)

### Community 5 - "test_layout.py"

Cohesion: 0.09
Nodes (28): \_imported_packages(), \_package_of(), Path, test_every_package_is_in_the_table(), test_imports_follow_the_table(), ast, base_ref(), block() (+20 more)

### Community 6 - "ag_news.py"

Cohesion: 0.13
Nodes (22): \_download_csv(), get_ag_news_dataloader(), get_bpe(), inspect_ag_news_dataset(), \_load_splits(), DataLoader, Path, ag_news.py Load and inspect a sampled AG News dataset, tokenized with our from-… (+14 more)

### Community 7 - "MaxHeap"

Cohesion: 0.11
Nodes (18): MaxHeap, PrioritizedItem, Binary max-heap of (priority, item) pairs with O(log n) priority updates.…, Return the current priority of an existing item in O(1). Args: item: Item…, One heap entry. Frozen so callers cannot change a priority behind the heap's…, Raise or lower the priority of an existing item in O(log n). Exactly one of…, Swap entry i with its parent until the invariant holds. Keeps the index map in…, Swap entry i with its larger child until the invariant holds. Keeps the index… (+10 more)

### Community 8 - "Getting started"

Cohesion: 0.09
Nodes (25): ConvolutionalMNISTClassifier, DeepMNISTClassifier, MNISTClassifier, Float, jaxtyped, NUM_CLASSES, Tensor, Return logits for `x`. (+17 more)

### Community 9 - "bpe.py"

Cohesion: 0.15
Nodes (14): \_base_vocab(), \_count_pairs(), PhraseNode, Byte-level Byte Pair Encoding: fit once, encode/decode many times. BPE turns…, One adjacent pair of tokens in the training text; internal to `train`. `phrase`…, Link adjacent token pairs into one doubly linked list per chunk, by pair. Node…, # TODO: make it training time specific, Rebuild (merges, vocab) as train returns them from each merge's id pair. Args:… (+6 more)

### Community 10 - "mnist.py"

Cohesion: 0.15
Nodes (19): accumulate_metrics(), calculate_metrics(), Tensor, Add this batch into `acc` in place. `acc.loss` holds the batch-size-weighted…, Return the epoch mean loss from `acc`, or 0.0 if nothing was accumulated., Return this batch's metrics; the default adds nothing beyond loss and count., reduce_metrics(), get_mnist_dataloader() (+11 more)

### Community 11 - "plot_attention_heads"

Cohesion: 0.17
Nodes (12): plot_attention_heads(), B, Float, H, Int, jaxtyped, L, Path (+4 more)

### Community 12 - "TrainState"

Cohesion: 0.13
Nodes (31): collect_system_metrics(), environment.py -- snapshot of host resource usage., Return current CPU/RAM usage and, if available, GPU memory stats., EpochRecord, EpochSpec, model.py -- dataclasses shared by the training loop., SystemMetrics, TrainState (+23 more)

### Community 13 - "test_bpe.py"

Cohesion: 0.10
Nodes (4): test_phrase_requires_encoding_xor_merge(), test_phrase_to_str_leaf(), test_phrase_to_str_merge_concatenates_children(), training_common_bpe

### Community 14 - "multi30ken_predictor.py"

Cohesion: 0.11
Nodes (27): Protected token ids for the byte-level BPE tokenizer in bpe.py. Ids 0-255 are…, ag_news_classifier.py Text classifier for the AG News dataset (see…, Multi30kEnPredictor, # TODO: Write one from scratch, # TODO: Implement loss function, # TODO: Implement beam/grid search, multi30ken_predictor.py Masked-token predictor for Multi30k English sentences…, Embedding → sin/cos positions → pre-norm encoder stack → vocab logits per… (+19 more)

### Community 15 - "Dataload"

Cohesion: 0.09
Nodes (23): A dataset returns one sample, the loader returns batches, A Dataset: the only required class, A Sampler: only for custom ordering, collate_fn: a function, not a class, Dataload, Each split gets its own loader, How a DataLoader works, How test leaks (+15 more)

### Community 16 - "padding_keep_mask"

Cohesion: 0.16
Nodes (18): causal_keep_mask(), pad_masked_mean(), padding_keep_mask(), B, Bool, D_MODEL, device, Float (+10 more)

### Community 17 - "hooks.py"

Cohesion: 0.14
Nodes (16): File, Files, mkdocs_config_defaults, mkdocs_structure_files, mkdocs_structure_pages, MkDocsConfig, Page, on_files() (+8 more)

### Community 18 - "test_mask.py"

Cohesion: 0.20
Nodes (9): test_attention_weights_sum_to_one(), test_matches_torch_reference(), test_causal_keep_mask_is_lower_triangular(), test_causal_mask_matches_torch_reference(), test_pad_masked_mean_all_padding_row_is_zeros_not_nan(), test_padding_keep_mask_values_and_shape(), torch_nn_functional, training_transformer_functions (+1 more)

### Community 19 - ".forward"

Cohesion: 0.22
Nodes (8): B, Float, Int, jaxtyped, L, NUM_CLASSES, Tensor, Return logits for a batch of right-padded token ids. Args: token_ids: Padded…

### Community 20 - "test_ag_news_classifier.py"

Cohesion: 0.17
Nodes (7): training_common, training_common_constants, training_dataload_ag_news, training_dataload_constants, training_model_ag_news_classifier, training_train_train_loop, training_transformer_probes

### Community 21 - "Shapes and Broadcasting — Q&A"

Cohesion: 0.05
Nodes (39): 1-D operands, 1. The comma is Python syntax, 2. What the tuple says, 3. Intuition: how many indices to reach one value, 4. Why it matters: `(3,)`, `(1, 3)` and `(3, 1)` are different tensors, Check your understanding, Common confusions, Habits that end guess-and-check (+31 more)

### Community 22 - "multi30ken.py"

Cohesion: 0.09
Nodes (29): Phrase, A token, as a binary tree of the merges that built it. A leaf sets `encoding`…, Raises ValueError unless exactly one of `encoding` or `first`+`second` is set., Render this phrase as a string, resolving merges recursively., \_get_cached_or_downloaded_txts(), get_multi30ken_mlm_dataloader(), \_load_merges(), Multi30kENPaths (+21 more)

### Community 23 - "HasAttentionMap"

Cohesion: 0.67
Nodes (3): HasAttentionMap, A model whose `forward` stores each block's (B, h, L, L) weights in block order., Protocol

### Community 24 - ".forward"

Cohesion: 0.26
Nodes (11): B, Bool, D_MODEL, Float, H, jaxtyped, L, Tensor (+3 more)

### Community 25 - "The Matrix Calculus You Need for Deep Learning"

Cohesion: 0.07
Nodes (28): 10. Resources, 1. Introduction, 2. Review: Scalar Derivative Rules, 3. Introduction to Vector Calculus and Partial Derivatives, 4.1 Generalization of the Jacobian, 4.2 Derivatives of Vector Element-wise Binary Operators, 4.3 Derivatives Involving Scalar Expansion, 4.4 Vector Sum Reduction (+20 more)

### Community 26 - "fit"

Cohesion: 0.05
Nodes (79): GitHub Pages workflow, eval_step(), \_extract_profiler_metrics(), fit(), load(), profiled_fit(), \_profiled_run_epoch(), Any (+71 more)

### Community 27 - ".create"

Cohesion: 0.29
Nodes (6): device, Module, Build a TrainState, moving `model` to `device` in place., test_train_state_create_moves_model_to_device(), LRScheduler, Optimizer

### Community 28 - "test_multi30ken.py"

Cohesion: 0.09
Nodes (29): bpe_lengths(), log_stats(), main(), ndarray, multi30ken_bpe_len.py Print BPE length stats (mean, min, max, percentiles) for…, Return the number of non-PAD_ID ids in each row of a Multi30kENDataset. Raises:…, Log count, mean, min, max, and PERCENTILES of one split's BPE lengths., Load every row of each Multi30k EN split and log its BPE length stats. (+21 more)

### Community 29 - "test_metrics.py"

Cohesion: 0.25
Nodes (7): test_accumulate_metrics_accumulates_across_batches(), test_accumulate_metrics_adds_weighted_loss(), test_calculate_metrics_returns_empty_batch_metrics(), test_reduce_metrics_returns_mean(), test_reduce_metrics_returns_zero_when_empty(), training_metrics_loss_only_metrics, training_train_model

### Community 30 - "Transformer Compute and Memory"

Cohesion: 0.17
Nodes (12): A Worked Training Budget, Activations grow with the microbatch, Common Confusions, Count the Training Work, How Data, Tensor, and Pipeline Parallelism Combine, Inference Memory: Weights and the KV Cache, Parameters and tokens are different choices, Recall Check (+4 more)

### Community 31 - "AGNewsClassifier"

Cohesion: 0.47
Nodes (6): AGNewsClassifier, Embedding + sin/cos positions → 2 pre-norm encoder blocks → mean pool → 4…, \_padded_ids(), test_classifier_forward_stores_detached_attention_map_per_block(), test_classifier_is_sensitive_to_token_order(), test_save_attention_maps_writes_first_batch_rows_once_per_epoch()

### Community 32 - "Random Streams and DataLoader Workers — Q&A"

Cohesion: 0.06
Nodes (34): A draw advances only the generator it reads, A toy generator, An unseeded generator always starts from the same seed, Basic usage, Cheat-Sheet, Check your understanding, Common confusions, Dropping `index` makes it worse (+26 more)

### Community 33 - "train"

Cohesion: 0.18
Nodes (17): decode(), encode(), \_pretokenize(), Learn up to num_merges BPE merges from text. Whitespace is normalized to EOW,…, Tokenize text with merges learned by train. Within each chunk, repeatedly…, Turn token ids back into text. Lossy: whitespace comes back as EOW, since train…, Replace every whitespace char with EOW and split into \_CHUNK_PATTERN chunks.…, train() (+9 more)

### Community 34 - "Concepts"

Cohesion: 0.08
Nodes (24): A batch from input to update, A gradient back through the model, Attention and gradient flow, Autograd and backprop, Browse by Topic, Concepts, Data, Data order and training behavior (+16 more)

### Community 35 - "ClassificationMetrics"

Cohesion: 0.15
Nodes (19): accumulate_metrics(), calculate_metrics(), ClassificationMetrics, \_macro_average(), Tensor, Count correct predictions and per-class TP/FP/FN in this batch. Args:…, Add this batch into `acc` in place. `acc.loss` holds the batch-size-weighted…, Return the mean over classes of hits / (hits + misses), counting 0/0 as 0.0. (+11 more)

### Community 36 - "train_config"

Cohesion: 0.10
Nodes (27): accumulate_metrics(), calculate_metrics(), MaskedTokenMetrics, Tensor, Count scored positions and correct top-1 predictions among them. Args:…, Add this batch into `acc` in place. `acc.loss` holds the loss sum over scored…, Return the per-position mean loss, accuracy, and perplexity for the epoch.…, reduce_metrics() (+19 more)

### Community 37 - "inspect_mnist_dataset"

Cohesion: 0.67
Nodes (3): inspect_mnist_dataset(), DataLoader, Print shape, class info, and one-batch tensor statistics. Call this before…

### Community 41 - "DataLoader — Batching and the Dataset Abstraction"

Cohesion: 0.13
Nodes (15): Check your understanding, Common confusions, Compose chains them, Computing Normalization Stats, DataLoader — Batching and the Dataset Abstraction, Format conversion — `ToTensor()`, How a Batch Becomes a Scalar Loss, How Augmentations Work (+7 more)

### Community 48 - "get_device"

Cohesion: 0.67
Nodes (3): get_device(), device, Return cuda or mps if available, else cpu. Set the `DISABLE_GPU=1` environment…

### Community 53 - "\_encode_rows"

Cohesion: 0.18
Nodes (10): AGNewsDataset, \_encode_rows(), Dataset, Tensor, Tokenize `rows` into a fixed-length AGNewsDataset, right-padded with PAD_ID., Pre-tokenized (ids, label) pairs; ids is (N, max_len), truncated and right-…, Store pre-tokenized ids and labels; see class docstring for shapes., Return the (ids, label) pair at `index`. (+2 more)

### Community 54 - ".forward"

Cohesion: 0.22
Nodes (8): B, Float, Int, jaxtyped, L, Tensor, Return logits for every position of a batch of masked, right-padded ids. Args:…, V

### Community 57 - "bpe_seq_len.py"

Cohesion: 0.14
Nodes (16): \_load_merges(), Store each merge as its (first_id, second_id) pair, which is all encode needs., Rebuild (merges, vocab) as bpe.train returns them; None if missing or stale., All (text, label) rows of a raw CSV; text is title + description, labels are…, \_read_raw_rows(), \_save_merges(), \_load_or_train_pairs(), main() (+8 more)

### Community 59 - "save_attention_maps"

Cohesion: 0.33
Nodes (6): Module, Path, Tensor, Save the first batch's attention maps for one epoch to `out_dir`. A probe for…, save_attention_maps(), test_save_attention_maps_rejects_model_without_attention_map()

## Knowledge Gaps

- **180 isolated node(s):** `dawnshard`, `graphify`, `Named-argument linting`, `Docstring linting`, `What does not go in a module docstring` (+175 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 471 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (\<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions

_Questions this graph is uniquely positioned to answer:_

- **Why does `Dawnshard` connect `fit` to `Getting started`, `index.md`, `TrainState`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Why does `fit()` connect `fit` to `SinusoidalEmbedding`, `ClassificationMetrics`, `train_config`, `Concepts`, `ag_news.py`, `Getting started`, `mnist.py`, `TrainState`, `multi30ken_predictor.py`, `Dataload`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `fit()` (e.g. with `Three splits, three jobs` and `EpochSpec`) actually correct?**
  _`fit()` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `TrainState` (e.g. with `serialize_train_state()` and `eval_step()`) actually correct?**
  _`TrainState` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `EpochSpec` (e.g. with `serialize_epoch_spec()` and `eval_step()`) actually correct?**
  _`EpochSpec` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `Metrics` (e.g. with `accumulate_metrics()` and `main()`) actually correct?**
  _`Metrics` has 16 INFERRED edges - model-reasoned connections that need verification._
- **What connects `dawnshard`, `graphify`, `Named-argument linting` to the rest of the system?**
  _180 weakly-connected nodes found - possible documentation gaps or missing edges._
