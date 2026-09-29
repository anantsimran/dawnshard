# Graph Report - dawnshard  (2026-09-29)

## Corpus Check
- 83 files · ~95,276 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: (none) 6, .css 1, .lock 1)

## Summary
- 1010 nodes · 1854 edges · 63 communities (45 shown, 18 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 313 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `dfd88b4b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Dawnshard
- index.md
- SinusoidalEmbedding
- attention
- compare_runs.py
- test_layout.py
- get_ag_news_dataloader
- MaxHeap
- Getting started
- bpe.py
- Metrics
- plot_attention_heads
- TrainState
- test_bpe.py
- multi30ken_predictor.py
- Dataload
- padding_keep_mask
- hooks.py
- MultiHeadAttentionLayer
- .forward
- test_ag_news_classifier.py
- Shapes and Broadcasting — Q&A
- Phrase
- HasAttentionMap
- .forward
- The Matrix Calculus You Need for Deep Learning
- fit
- ag_news.py
- test_multi30ken.py
- test_train_loop.py
- Transformer Compute and Memory
- CLAUDE.md
- Random Streams and DataLoader Workers — Q&A
- BPE tokenizer
- Concepts
- ClassificationMetrics
- train_config
- inspect_mnist_dataset
- pre-commit
- main
- DataLoader — Batching and the Dataset Abstraction
- Sunrise favicon
- dawnshard
- Training run metrics plot
- get_device
- _get_cached_or_downloaded_txts
- AGENTS.md
- inspect_ag_news_dataset
- MathJax page override
- _encode_rows
- B
- Any
- DataLoader
- _read_raw_rows
- Float
- Int
- jaxtyped
- L
- Tensor

## God Nodes (most connected - your core abstractions)
1. `fit()` - 47 edges
2. `TrainState` - 35 edges
3. `EpochSpec` - 30 edges
4. `Metrics` - 26 edges
5. `profiled_fit()` - 22 edges
6. `attention()` - 22 edges
7. `Phrase` - 20 edges
8. `eval_step()` - 20 edges
9. `run_epoch()` - 20 edges
10. `make_state_and_config()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Probes: a function of the training state` --references--> `EpochSpec`  [INFERRED]
  README.md → app/src/training/train/model.py
- `Every rule has a tool behind it` --references--> `fit()`  [INFERRED]
  README.md → app/src/training/train/train_loop.py
- `When the numbers aren't enough` --references--> `profiled_fit()`  [INFERRED]
  README.md → app/src/training/train/train_loop.py
- `Tutorials site` --references--> `EpochSpec`  [INFERRED]
  CLAUDE.md → app/src/training/train/model.py
- `Diagram rules` --references--> `pad_masked_mean()`  [INFERRED]
  CLAUDE.md → app/src/training/transformer/mask.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Training components communicate through shared state** — tutorials_pytorch_nn_module_and_multi_layer_networks_module_parameter_registration, tutorials_training_optimizer_and_scheduler_shared_model_optimizer_parameters, tutorials_training_optimizer_and_scheduler_scheduler_edits_optimizer_learning_rate, tutorials_training_training_loop_five_step_training_loop [INFERRED 0.95]

## Communities (63 total, 18 thin omitted)

### Community 0 - "Dawnshard"
Cohesion: 0.11
Nodes (19): GitHub Pages workflow, MkDocs site configuration, Landing page template, Compare two runs, Dawnshard, Debuggability, Every rule has a tool behind it, Exactly one place for everything (+11 more)

### Community 1 - "index.md"
Cohesion: 0.06
Nodes (57): Autograd computation graph, Differentiability and subgradients, Grad And Descent, Leaf gradient accumulation, Autograd graph visualization, Inspecting Your Model Best Practices, Model shape contracts, Module hooks (+49 more)

### Community 2 - "SinusoidalEmbedding"
Cohesion: 0.14
Nodes (16): Build the embedding, encoder stack, and classifier head. Args: vocab_size:…, Build the embedding, position table, encoder stack, final norm, and head. Args:…, Add fixed sinusoidal position encodings to token embeddings. Attention is…, SinusoidalEmbedding, test_each_feature_pair_lies_on_the_unit_circle(), test_every_position_gets_a_different_encoding(), test_first_position_is_sin_zero_and_cos_zero(), test_forward_adds_the_table_to_every_row_of_the_batch() (+8 more)

### Community 3 - "attention"
Cohesion: 0.14
Nodes (14): attention(), B, Bool, Float, H, jaxtyped, L, Tensor (+6 more)

### Community 4 - "compare_runs.py"
Cohesion: 0.08
Nodes (31): load_history(), main(), Figure, Path, Compare per-epoch metrics from two training runs on the same graph., Write `fig` as a base64-embedded PNG in an HTML file and open it in the browser., Load per-epoch records, accepting either a bare JSON list or `{"history":…, CLI entry point: plot shared metrics from two history files and open the… (+23 more)

### Community 5 - "test_layout.py"
Cohesion: 0.09
Nodes (28): _imported_packages(), _package_of(), Path, test_every_package_is_in_the_table(), test_imports_follow_the_table(), ast, base_ref(), block() (+20 more)

### Community 6 - "get_ag_news_dataloader"
Cohesion: 0.17
Nodes (17): _download_csv(), get_ag_news_dataloader(), get_bpe(), _load_splits(), Path, Store each merge as its (first_id, second_id) pair, which is all encode needs., Load the sampled AG News splits, train BPE on the train texts, and return…, Return (merges, vocab) trained on the train split, loading cached merges if… (+9 more)

### Community 7 - "MaxHeap"
Cohesion: 0.07
Nodes (33): MaxHeap, PrioritizedItem, Binary max-heap of (priority, item) pairs with O(log n) priority updates.…, Return the current priority of an existing item in O(1). Args: item: Item…, One heap entry. Frozen so callers cannot change a priority behind the heap's…, Raise or lower the priority of an existing item in O(log n). Exactly one of…, Swap entry i with its parent until the invariant holds. Keeps the index map in…, Swap entry i with its larger child until the invariant holds. Keeps the index… (+25 more)

### Community 8 - "Getting started"
Cohesion: 0.09
Nodes (24): ConvolutionalMNISTClassifier, DeepMNISTClassifier, MNISTClassifier, Float, jaxtyped, NUM_CLASSES, Tensor, Return logits for `x`. (+16 more)

### Community 9 - "bpe.py"
Cohesion: 0.12
Nodes (20): _base_vocab(), _count_pairs(), PhraseNode, _pretokenize(), Byte-level Byte Pair Encoding: fit once, encode/decode many times. BPE turns…, One adjacent pair of tokens in the training text; internal to `train`. `phrase`…, Link adjacent token pairs into one doubly linked list per chunk, by pair. Node…, # TODO: make it training time specific (+12 more)

### Community 10 - "Metrics"
Cohesion: 0.17
Nodes (15): accumulate_metrics(), calculate_metrics(), Tensor, Add this batch into `acc` in place. `acc.loss` holds the batch-size-weighted…, Return the epoch mean loss from `acc`, or 0.0 if nothing was accumulated., Return this batch's metrics; the default adds nothing beyond loss and count., reduce_metrics(), get_mnist_dataloader() (+7 more)

### Community 11 - "plot_attention_heads"
Cohesion: 0.14
Nodes (15): plot_attention_heads(), plot_attention_probes(), B, Float, H, Int, jaxtyped, L (+7 more)

### Community 12 - "TrainState"
Cohesion: 0.11
Nodes (37): collect_system_metrics(), environment.py -- snapshot of host resource usage., Return current CPU/RAM usage and, if available, GPU memory stats., EpochRecord, EpochSpec, model.py -- dataclasses shared by the training loop., SystemMetrics, TrainState (+29 more)

### Community 13 - "test_bpe.py"
Cohesion: 0.10
Nodes (4): test_phrase_requires_encoding_xor_merge(), test_phrase_to_str_leaf(), test_phrase_to_str_merge_concatenates_children(), training_common_bpe

### Community 14 - "multi30ken_predictor.py"
Cohesion: 0.11
Nodes (30): Protected token ids for the byte-level BPE tokenizer in bpe.py. Ids 0-255 are…, multi30ken.py Multi30k English sentences as masked-language-modelling batches,…, ag_news_classifier.py Text classifier for the AG News dataset (see…, mnist_classifier.py Load, inspect, train, and evaluate on the MNIST dataset…, # TODO: Write one from scratch, # TODO: Implement loss function, # TODO: Implement beam/grid search, multi30ken_predictor.py Masked-token predictor for Multi30k English sentences… (+22 more)

### Community 15 - "Dataload"
Cohesion: 0.09
Nodes (23): A dataset returns one sample, the loader returns batches, A Dataset: the only required class, A Sampler: only for custom ordering, collate_fn: a function, not a class, Dataload, Each split gets its own loader, How a DataLoader works, How test leaks (+15 more)

### Community 16 - "padding_keep_mask"
Cohesion: 0.13
Nodes (24): causal_keep_mask(), pad_masked_mean(), padding_keep_mask(), B, Bool, D_MODEL, device, Float (+16 more)

### Community 17 - "hooks.py"
Cohesion: 0.14
Nodes (16): File, Files, mkdocs_config_defaults, mkdocs_structure_files, mkdocs_structure_pages, MkDocsConfig, Page, on_files() (+8 more)

### Community 18 - "MultiHeadAttentionLayer"
Cohesion: 0.16
Nodes (13): EncoderBlock, MultiHeadAttentionLayer, Multi-head self-attention for an encoder, with an optional padding mask.…, Raises ValueError if `d_model` is not divisible by `h`. Args: qk_norm: RMSNorm…, One pre-norm encoder block: self-attention, then a position-wise FFN. Each…, Build the block's two sublayers and their norms. Args: qk_norm: Forwarded to…, Precompute the (seq_length, d_model) position table. Args: seq_length: Longest…, Attention (+5 more)

### Community 19 - ".forward"
Cohesion: 0.15
Nodes (12): B, Float, Int, jaxtyped, L, NUM_CLASSES, Tensor, Return logits for a batch of right-padded token ids. Args: token_ids: Padded… (+4 more)

### Community 20 - "test_ag_news_classifier.py"
Cohesion: 0.11
Nodes (19): AGNewsClassifier, Embedding + sin/cos positions → 2 pre-norm encoder blocks → mean pool → 4…, Module, Path, Tensor, Save the first batch's attention maps for one epoch to `out_dir`. A probe for…, save_attention_maps(), _padded_ids() (+11 more)

### Community 21 - "Shapes and Broadcasting — Q&A"
Cohesion: 0.05
Nodes (39): 1-D operands, 1. The comma is Python syntax, 2. What the tuple says, 3. Intuition: how many indices to reach one value, 4. Why it matters: `(3,)`, `(1, 3)` and `(3, 1)` are different tensors, Check your understanding, Common confusions, Habits that end guess-and-check (+31 more)

### Community 22 - "Phrase"
Cohesion: 0.15
Nodes (16): Phrase, A token, as a binary tree of the merges that built it. A leaf sets `encoding`…, Raises ValueError unless exactly one of `encoding` or `first`+`second` is set., Render this phrase as a string, resolving merges recursively., get_multi30ken_mlm_dataloader(), _load_merges(), DataLoader, Path (+8 more)

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
Cohesion: 0.15
Nodes (31): eval_step(), _extract_profiler_metrics(), fit(), profiled_fit(), _profiled_run_epoch(), Any, DataLoader, Path (+23 more)

### Community 27 - "ag_news.py"
Cohesion: 0.22
Nodes (10): _load_merges(), ag_news.py Load and inspect a sampled AG News dataset, tokenized with our from-…, Rebuild (merges, vocab) as bpe.train returns them; None if missing or stale., _load_or_train_pairs(), bpe_seq_len.py Plot mean BPE tokens per sample vs num_merges on the full AG…, Return MAX_MERGES merges as (first_id, second_id) pairs, training on a cache…, csv, json (+2 more)

### Community 28 - "test_multi30ken.py"
Cohesion: 0.08
Nodes (32): bpe_lengths(), log_stats(), main(), ndarray, multi30ken_bpe_len.py Print BPE length stats (mean, min, max, percentiles) for…, Return the number of non-PAD_ID ids in each row of a Multi30kENDataset. Raises:…, Log count, mean, min, max, and PERCENTILES of one split's BPE lengths., Load every row of each Multi30k EN split and log its BPE length stats. (+24 more)

### Community 29 - "test_train_loop.py"
Cohesion: 0.08
Nodes (32): device, Module, Build a TrainState, moving `model` to `device` in place., test_accumulate_metrics_accumulates_across_batches(), test_accumulate_metrics_adds_weighted_loss(), test_calculate_metrics_returns_empty_batch_metrics(), test_reduce_metrics_returns_mean(), test_reduce_metrics_returns_zero_when_empty() (+24 more)

### Community 30 - "Transformer Compute and Memory"
Cohesion: 0.17
Nodes (12): A Worked Training Budget, Activations grow with the microbatch, Common Confusions, Count the Training Work, How Data, Tensor, and Pipeline Parallelism Combine, Inference Memory: Weights and the KV Cache, Parameters and tokens are different choices, Recall Check (+4 more)

### Community 31 - "CLAUDE.md"
Cohesion: 0.15
Nodes (11): Anchors other files depend on, Before creating a PR, Docstring linting, Line length, Named-argument linting, Pre-commit checks, README structure, Style (+3 more)

### Community 32 - "Random Streams and DataLoader Workers — Q&A"
Cohesion: 0.06
Nodes (34): A draw advances only the generator it reads, A toy generator, An unseeded generator always starts from the same seed, Basic usage, Cheat-Sheet, Check your understanding, Common confusions, Dropping `index` makes it worse (+26 more)

### Community 33 - "BPE tokenizer"
Cohesion: 0.21
Nodes (13): decode(), encode(), Tokenize text with merges learned by train. Within each chunk, repeatedly…, Turn token ids back into text. Lossy: whitespace comes back as EOW, since train…, Batch with padding, BPE tokenizer, Common, Encode and decode (+5 more)

### Community 34 - "Concepts"
Cohesion: 0.08
Nodes (24): A batch from input to update, A gradient back through the model, Attention and gradient flow, Autograd and backprop, Browse by Topic, Concepts, Data, Data order and training behavior (+16 more)

### Community 35 - "ClassificationMetrics"
Cohesion: 0.25
Nodes (10): calculate_metrics(), ClassificationMetrics, Tensor, Count correct predictions and per-class TP/FP/FN in this batch. Args:…, test_accumulate_metrics_accumulates_across_batches(), test_calculate_metrics_counts_per_class(), test_reduce_metrics_counts_unseen_class_as_zero(), test_reduce_metrics_returns_macro_precision_and_recall() (+2 more)

### Community 36 - "train_config"
Cohesion: 0.06
Nodes (37): Any, accumulate_metrics(), calculate_metrics(), MaskedTokenMetrics, Tensor, Count scored positions and correct top-1 predictions among them. Args:…, Add this batch into `acc` in place. `acc.loss` holds the loss sum over scored…, Return the per-position mean loss, accuracy, and perplexity for the epoch.… (+29 more)

### Community 37 - "inspect_mnist_dataset"
Cohesion: 0.67
Nodes (3): inspect_mnist_dataset(), DataLoader, Print shape, class info, and one-batch tensor statistics. Call this before…

### Community 40 - "main"
Cohesion: 0.20
Nodes (11): accumulate_metrics(), _macro_average(), Add this batch into `acc` in place. `acc.loss` holds the batch-size-weighted…, Return the mean over classes of hits / (hits + misses), counting 0/0 as 0.0., Return epoch mean loss, accuracy, and macro-averaged precision and recall.…, reduce_metrics(), main(), Train AGNewsClassifier on AG News, optionally logging to Weights & Biases. (+3 more)

### Community 41 - "DataLoader — Batching and the Dataset Abstraction"
Cohesion: 0.13
Nodes (15): Check your understanding, Common confusions, Compose chains them, Computing Normalization Stats, DataLoader — Batching and the Dataset Abstraction, Format conversion — `ToTensor()`, How a Batch Becomes a Scalar Loss, How Augmentations Work (+7 more)

### Community 48 - "get_device"
Cohesion: 0.67
Nodes (3): get_device(), device, Return cuda or mps if available, else cpu. Set the `DISABLE_GPU=1` environment…

### Community 49 - "_get_cached_or_downloaded_txts"
Cohesion: 0.25
Nodes (7): _get_cached_or_downloaded_txts(), Multi30kENPaths, Paths of the three split text files, one English sentence per line., Return the paths of the train, val, and test text files. On a cache miss,…, log_elapsed(), Log how long the `with` block took, as "<label> in 1.2s". Args: label: What the…, contextlib

### Community 51 - "inspect_ag_news_dataset"
Cohesion: 0.67
Nodes (3): inspect_ag_news_dataset(), DataLoader, Print size, class balance, and one-batch tensor statistics. Call this before…

### Community 53 - "_encode_rows"
Cohesion: 0.18
Nodes (10): AGNewsDataset, _encode_rows(), Dataset, Tensor, Tokenize `rows` into a fixed-length AGNewsDataset, right-padded with PAD_ID., Pre-tokenized (ids, label) pairs; ids is (N, max_len), truncated and right-…, Store pre-tokenized ids and labels; see class docstring for shapes., Return the (ids, label) pair at `index`. (+2 more)

### Community 57 - "_read_raw_rows"
Cohesion: 0.22
Nodes (9): All (text, label) rows of a raw CSV; text is title + description, labels are…, _read_raw_rows(), main(), merge_counts(), ndarray, counts[r] = how many times merge r fires when each text is encoded on its own,…, Compute and plot mean BPE sequence length vs num_merges, writing data and plot…, test_read_raw_rows_converts_label_to_zero_indexed_and_joins_title_and_description() (+1 more)

## Knowledge Gaps
- **180 isolated node(s):** `Why the name`, `Metrics: a function of the outputs`, `No hidden complexity`, `The terminal: one line per epoch`, `Plot it locally` (+175 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 480 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Dawnshard` connect `Dawnshard` to `Getting started`, `index.md`, `fit`, `TrainState`?**
  _High betweenness centrality (0.174) - this node is a cross-community bridge._
- **Why does `fit()` connect `fit` to `Dawnshard`, `Concepts`, `train_config`, `get_ag_news_dataloader`, `main`, `Metrics`, `plot_attention_heads`, `TrainState`, `multi30ken_predictor.py`, `Dataload`, `test_train_loop.py`, `CLAUDE.md`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Why does `Theory & Revision` connect `Concepts` to `index.md`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `fit()` (e.g. with `Three splits, three jobs` and `EpochSpec`) actually correct?**
  _`fit()` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `TrainState` (e.g. with `serialize_train_state()` and `eval_step()`) actually correct?**
  _`TrainState` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `EpochSpec` (e.g. with `serialize_epoch_spec()` and `eval_step()`) actually correct?**
  _`EpochSpec` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `Metrics` (e.g. with `accumulate_metrics()` and `main()`) actually correct?**
  _`Metrics` has 16 INFERRED edges - model-reasoned connections that need verification._