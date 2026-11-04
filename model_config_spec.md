# Model config and self-describing checkpoints

**Status:** draft for review, 2026-11-03. The decisions in §3 are open. Every
requirement is written against the default answer, and §3 says what changes if the
other answer is picked.

**In one line:** a trained `Multi30kTranslator` is saved as one file that is enough to
rebuild it, and that file translates a sentence from the command line.

The format decision itself is recorded in
[ADR 0001](docs/adr/0001-self-describing-checkpoints.md). The reasoning behind it is in
[docs/discussions/](docs/discussions/checkpoint-pickle-vs-state-dict.md).

Requirement IDs (`CFG-2`, `LOAD-4`) are stable, so review comments can cite them. Every
requirement is a "must" except the one marked "should".

______________________________________________________________________

## 1. Problem

Translating one sentence with `app/history/checkpoints/multi30k_translator_best.pt`
needs three things the file does not hold.

1. **The constructor arguments.** [save_state](app/src/training/train/utils.py#L14)
   writes only `state_dict`s, and [load](app/src/training/train/train_loop.py#L135)
   restores them into a `TrainState` the caller already built. The arguments behind
   the best checkpoint exist only as
   [BEST_CONFIG](app/src/training/experiments/multi30k_translator_ablation.py#L180) in
   the ablation module. The history file records the class name and every parameter's
   shape ([serialization.py:10](app/src/training/train/serialization.py#L10)): enough
   to tell two runs apart, not enough to rebuild either.
1. **The tokenizers.** Embedding rows are indexed by BPE ids, and an id means whatever
   the merges file named by `tokenizer_path.txt` says at load time
   ([multi30k.py:178](app/src/training/dataload/multi30k.py#L178)). The translator's
   `main()` retrains both tokenizers and sets them as the default
   ([multi30k_translator.py:315](app/src/training/model/multi30k_translator.py#L315)),
   so every run repoints that file. Nothing records which merges a checkpoint was
   trained with.
1. **A decode loop.** The translator only runs under teacher forcing. No function turns
   a source sentence into a target sentence; the file opens with a TODO for one
   ([multi30k_translator.py:2](app/src/training/model/multi30k_translator.py#L2)).

The architecture itself lives in module constants
([multi30k_translator.py:54-71](app/src/training/model/multi30k_translator.py#L54-L71))
that `main()` wires into the constructor by hand. The ablation's
[Config](app/src/training/experiments/multi30k_translator_ablation.py#L74) describes
the same model a second time, with different field names.

______________________________________________________________________

## 2. Goals and non-goals

### Goals

- **G1.** A checkpoint file alone rebuilds the trained model and both tokenizers, on a
  machine with no `app/datasets/` and no `tokenizer_path.txt`.
- **G2.** One typed config object describes the translator's architecture. `main()`,
  saving, and loading all go through it, and the mapping from config to constructor
  exists in one place.
- **G3.** A saved translator translates a sentence from the command line.
- **G4.** `train/` stays model-agnostic. It stores what the model module hands it and
  interprets none of it.

### Non-goals

- **Ablation studies.** `app/src/training/experiments/` keeps its own `Config`,
  `build_train_state`, `BEST_CONFIG`, and `train_best`. Its one change is an import
  (CFG-6). Until a later change, two dataclasses describe the translator.
- **Converting old checkpoints.** `multi30k_translator_best.pt` stays in the old
  format. Converting it would mean guessing which merges it was trained with.
- **Best-epoch weights.** `fit` keeps no best-epoch weights, so `main()` saves the last
  epoch's. See §9.
- **Beam search and batched decoding.**
- **Other models** (D2), **training hyperparameters in the config** (D3), **config
  files and command-line overrides** (D4), **history-file changes** (D6).
- **`NUM_SRC_MERGES` and `NUM_TGT_MERGES`**
  ([multi30k_translator.py:58-59](app/src/training/model/multi30k_translator.py#L58-L59)).
  Nothing reads them; the loader uses `BPE_NUM_MERGES` from `dataload/multi30k.py`.
  Wiring or deleting them is a separate change.

______________________________________________________________________

## 3. Decisions

| ID | Question | Default (this spec) | Alternative | What the alternative changes |
|---|---|---|---|---|
| D1 | What does "run it" mean? | Translate one sentence, greedily | Score the checkpoint on the test split, or retrain from a saved config | Test split: §6.5 is replaced by an `evaluate` command that runs `run_epoch` with `eval_step` over the test loader. That path needs the dataset, so G1 shrinks to rebuilding the model. Retrain: needs D3's alternative, because an architecture alone does not say how a model was trained. |
| D2 | Which models? | `Multi30kTranslator` only | All three transformer models | A shared `ModelConfig` base in `train/` plus a name-to-class registry, so one `load_model(path)` opens any checkpoint. CFG, CKPT, LOAD, and TEST repeat for `AGNewsClassifier` and `Multi30kEnPredictor`. |
| D3 | What goes in the config? | Architecture only | Also learning rate, epochs, batch size, and schedule | A second frozen dataclass, `TrainConfig`, stored in the metadata beside `config`. `main()` reads both, and the training constants CFG-4 keeps move into it. |
| D4 | Config format | Python dataclass in code; a plain dict inside the checkpoint | YAML files passed with `--config` | `pyyaml` becomes a direct dependency (today it is in `uv.lock` only through wandb), plus a `configs/` folder and a command-line path that reads one. |
| D5 | Where does the tokenizer live? | Inside the checkpoint, as merge id pairs | The merges file's path | A smaller file, but G1 fails: the path points into `app/datasets/`, which is gitignored and not copied with the checkpoint. |
| D6 | Record the config in the history file too? | No | Yes | `fit` and `profiled_fit` gain an argument. That changes the README's one-screen run, the `home.html` hero, and the Using the loop sections. |
| D7 | Config field layout | One field per constructor parameter | Shared fields, as the ablation `Config` has them (`num_heads`, `ffn_multiplier`, `seq_len`) | `build_model` maps one field to two parameters, and the config can no longer describe an asymmetric model that the constructor accepts. |

______________________________________________________________________

## 4. Constraints the existing code fixes

These are facts about code that exists today. The requirements satisfy them; they are
not up for decision in this change.

- **C1. One stacked input.** `Multi30kTranslator.forward` takes one `(B, 2, L)` tensor,
  the source at index 0 and the target input at index 1
  ([multi30k_translator.py:226](app/src/training/model/multi30k_translator.py#L226)),
  because `train_step` calls `model(X)` with a single input. Source and target share
  `L` on every call, decoding included.
- **C2. The loader's row layout.** Source: `BOS_ID`, the tokens truncated to `L − 1`,
  then `PAD_ID`; no `EOS_ID`. Target input: `BOS_ID`, then the tokens. Target output:
  the tokens, then `EOS_ID`. Position `l` of the target input predicts position `l` of
  the target output ([multi30k.py:383-391](app/src/training/dataload/multi30k.py#L383-L391)).
- **C3. Flat logits.** `forward` returns `(B*L, V)`, and row `b*L + l` is target
  position `l` of pair `b`.
- **C4. Vocab size follows from the merges.** The base vocab holds the 256 bytes and
  all six protected ids, so `len(vocab) == FIRST_MERGE_ID + len(merges)`, with
  `FIRST_MERGE_ID == 262` ([bpe.py:137](app/src/training/common/bpe.py#L137)). The
  embedding sizes need not be stored separately.
- **C5. `torch.load` defaults to `weights_only=True`** (torch 2.12.1 in `uv.lock`).
  Tensors and plain containers of scalars load under it, and an arbitrary pickled
  object such as a dataclass instance raises. TEST-2 confirms the metadata loads.
- **C6. The import table.** `train/` may import only `constants` and `utils`, and
  `model/` may not import `experiments/`
  ([test_layout.py:10](app/tests/test_layout.py#L10)). `NORMS`, the norm-name map,
  lives in `experiments/common.py`
  ([common.py:37](app/src/training/experiments/common.py#L37)).
- **C7. Tokenizer round trip.** `bpe.tokenizer_from_pairs` rebuilds `(merges, vocab)`
  from merge id pairs ([bpe.py:313](app/src/training/common/bpe.py#L313)). Its inverse
  exists only inline, in `_save_merges`
  ([multi30k.py:127](app/src/training/dataload/multi30k.py#L127)).
- **C8. `load` reads named keys.** It reads `model`, `optimizer`, and `scheduler` when
  present ([train_loop.py:135](app/src/training/train/train_loop.py#L135)), so a new
  payload key does not affect it.
- **C9. Eval mode.** Dropout is active in train mode, so decoding runs in eval mode.
  Every `forward` overwrites `attention_map`, which is harmless here.
- **C10. `decode` is lossy.** It drops protected ids and returns all whitespace as
  single spaces ([bpe.py:419](app/src/training/common/bpe.py#L419)).

______________________________________________________________________

## 5. Design overview

The checkpoint keeps today's three `state_dict`s. It gains the fields that resuming a
run needs (epoch, seed, run id, RNG states) and one dict that holds everything specific
to the translator. Every value is a tensor, a number, a string, or a plain container of
those (CKPT-9):

```
multi30k_translator_<run_uuid>.pt
├── model        state_dict, as today
├── optimizer    state_dict, as today
├── scheduler    state_dict, as today, only when the run has one
├── epoch        int, the last completed epoch
├── train_seed   int
├── run_uuid     str, the same id that names the history file
├── rng_state
│   ├── torch    torch.get_rng_state()
│   ├── cuda     torch.cuda.get_rng_state_all(), when CUDA is in use
│   └── python   random.getstate()
└── metadata     new; train/ stores it and never reads it
    ├── model_class          "Multi30kTranslator"; the loader refuses anything else
    ├── config               dataclasses.asdict(Multi30kTranslatorConfig)
    ├── src_lang             Multi30k language code, e.g. "en"
    ├── tgt_lang             e.g. "de"
    ├── src_tokenizer_pairs  [[first_id, second_id], ...], in merge-rank order
    └── tgt_tokenizer_pairs  the same, for the target tokenizer
```

Where `config` sits (inside `metadata` or beside `model`) and which package writes the
training-run fields are open (§10). The tokenizer keys follow D5, which
[the tokenizer discussion](docs/discussions/bpe-tokenizer-in-checkpoint.md) still has
open.

The diagrams below show the single-save version, with one write at the end of
`main()`. ADR 0001 moves the write to between epochs, and how `fit` triggers it is open
(§10):

```
main()
  │
  ├─ get_multi30k_translation_dataloader ──► loaders, src and tgt merges and vocab
  │
  ├─ build_model(DEFAULT_CONFIG, len(src_vocab), len(tgt_vocab))
  │
  ├─ fit                                      writes app/history/<run_uuid>.json
  │
  └─ save_translator ──► save_state(..., metadata=...)
                              │
                              ▼
          app/history/checkpoints/multi30k_translator_<run_uuid>.pt
```

Loading reads nothing but that file:

```
multi30k_translator_<run_uuid>.pt
  │  torch.load(map_location=device); weights_only stays True
  ▼
load_translator
  ├─ metadata["src_tokenizer_pairs"] ──► tokenizer_from_pairs ──► src merges, vocab
  ├─ metadata["tgt_tokenizer_pairs"] ──► tokenizer_from_pairs ──► tgt merges, vocab
  ├─ metadata["config"] ──► build_model(config, len(src_vocab), len(tgt_vocab))
  └─ payload["model"] ──► load_state_dict(strict=True) ──► to(device), eval()
        │
        ▼  SavedTranslator
translate(saved, text) ──► greedy decode, at most L − 1 tokens ──► str
```

______________________________________________________________________

## 6. Requirements

### 6.1 Config (CFG)

- **CFG-1.** `model/multi30k_translator.py` defines `Multi30kTranslatorConfig`, a
  frozen dataclass.
- **CFG-2.** It has one field per `Multi30kTranslator.__init__` parameter, under the
  same name, except `src_vocab_size` and `tgt_vocab_size`, which come from the
  tokenizers (C4). The fields are `d_model`, `src_seq_length`, `tgt_seq_length`,
  `dropout`, `num_encoder_heads`, `num_decoder_heads`, `encoder_ffn_multiplier`,
  `decoder_ffn_multiplier`, `num_encoders`, `num_decoders`, `qk_norm`, and `norm`
  (D7).
- **CFG-3.** Every field is a JSON-native type. `norm` is
  `Literal["layernorm", "rmsnorm"]`, not a class.
- **CFG-4.** `DEFAULT_CONFIG` replaces the architecture constants `D_MODEL_SIZE`,
  `DROPOUT_RATE`, `NUM_ENCODERS`, `NUM_DECODERS`, `NUM_HEADS`, `FFN_MULTIPLIER`,
  `QK_NORM`, and `NORM`, with today's values. `SEQ_LEN` stays a constant because the
  loader reads it too, and it fills both sequence-length fields. The data and training
  constants (`SOURCE_LANG`, `TARGET_LANG`, `BATCH_SIZE`, `NUM_EPOCHS`,
  `LEARNING_RATE`, `COSINE_ANNEALING`, `PROBE_SENTENCES`) stay as they are (D3).
- **CFG-5.** `build_model(config, src_vocab_size, tgt_vocab_size)` returns a
  `Multi30kTranslator`. It is the only code that maps config fields to constructor
  arguments; `main()` and `load_translator` both call it.
- **CFG-6.** `NORMS` moves from `experiments/common.py` to `transformer/modules.py`,
  next to the blocks whose `norm` argument it names, and `experiments/common.py`
  imports it from there. No second copy exists, and the ablation code behaves as
  before.
- **CFG-7.** The config adds no validation of its own. The constructor's checks still
  apply: a `d_model` not divisible by a head count raises `ValueError` from the blocks.

### 6.2 Tokenizer serialization (TOK)

- **TOK-1.** `common/bpe.py` gains `pairs_from_merges(merges)`, which returns each
  merge's `(first_id, second_id)` in rank order. For any `merges, vocab` that
  `bpe.train` returns, `tokenizer_from_pairs(pairs=pairs_from_merges(merges=merges))`
  returns equal merges and vocab.
- **TOK-2.** `_save_merges` in `dataload/multi30k.py` calls `pairs_from_merges`
  instead of building the ids inline. Merges files keep their format byte for byte, so
  existing files still load.
- **TOK-3.** The paragraph in `bpe.py`'s module docstring that explains how to save a
  tokenizer names `pairs_from_merges`. It names no dataset.

### 6.3 Checkpoint (CKPT)

- **CKPT-1.** `save_state` gains `metadata: Optional[dict] = None`. When given, the
  payload holds it under `"metadata"`. When `None`, the payload is exactly today's.
- **CKPT-2.** `train/` never reads `metadata`. `save_state`'s docstring says it must
  hold only plain containers and scalars (C5).
- **CKPT-3.** `load` is unchanged, so a new-format checkpoint still resumes training
  through `load` and a hand-built `TrainState`.
- **CKPT-4.** The translator's metadata has exactly the keys in the §5 layout.
- **CKPT-5.** `save_translator` in `model/multi30k_translator.py` takes the
  `TrainState`, the config, both tokenizers' merges, both language codes, and the
  checkpoint path. It builds that dict and calls `save_state`. Like `save_state`, it
  does not create the parent folder.
- **CKPT-6.** `main()` saves one checkpoint after `fit` returns, to
  `CHECKPOINTS_PATH / f"multi30k_translator_{run_uuid}.pt"`, creating the folder first,
  and logs the path. `run_uuid` also names the history file, so every checkpoint pairs
  with the history of the run that produced it. ADR 0001 extends this to a checkpoint
  between epochs; see §10.
- **CKPT-7.** Every checkpoint holds the model `state_dict`, the optimizer `state_dict`,
  the scheduler `state_dict` when the run has a scheduler, the config as a plain dict
  (CFG-3), `epoch`, `train_seed`, `run_uuid`, and the torch, CUDA, and Python `random`
  RNG states. There is no separate weights-only file for inference. Running needs only
  the weights and the metadata, and the optimizer state is the cost of resuming.
- **CKPT-8.** The checkpoint holds no pickled class, module, or dataclass instance. The
  config goes in as `dataclasses.asdict(config)`, and `norm` is a string (CFG-3), so
  nothing in the file names a class that has to exist at load time.
- **CKPT-9.** A checkpoint loads under `torch.load`'s default `weights_only=True`. That
  includes the RNG states: the torch state is a byte tensor, and `random.getstate()`
  returns a tuple of ints and `None` or a float. TEST-8 covers the round trip.
- **CKPT-10.** The file is written with `torch.save` on the payload dict, not on the
  model.

### 6.4 Loading (LOAD)

- **LOAD-1.** `load_translator(checkpoint_path, device)` in
  `model/multi30k_translator.py` returns a frozen dataclass, `SavedTranslator`, holding
  the model, its config, both tokenizers' merges and vocab, and both language codes.
- **LOAD-2.** It calls `torch.load` with `map_location=device` and leaves
  `weights_only` at its default. It never passes `weights_only=False`.
- **LOAD-3.** It rebuilds both tokenizers with `tokenizer_from_pairs` and the config
  from `metadata["config"]`, then builds the model with `build_model`, sized by
  `len(src_vocab)` and `len(tgt_vocab)`. It loads the weights with `load_state_dict` in
  strict mode, moves the model to `device`, and puts it in eval mode.
- **LOAD-4.** It fails loudly:
  - No `"metadata"` key: `ValueError` saying the checkpoint predates configs and has to
    be retrained with `main()`.
  - `metadata["model_class"]` is not `"Multi30kTranslator"`: `ValueError` naming the
    class it found.
  - Weight names or shapes that do not match the config: PyTorch's own
    `load_state_dict` error, neither caught nor rewrapped.
- **LOAD-5.** It reads no file but the checkpoint: no dataset, no merges file, no
  `tokenizer_path.txt` (G1).
- **LOAD-6.** It builds no optimizer, criterion, or scheduler.
- **LOAD-7.** The saved device does not matter. `map_location` places the loaded
  tensors, and `load_state_dict` copies values into parameters that are already on the
  target device, so a checkpoint trained on a GPU loads on a CPU-only machine. The
  optimizer's `load_state_dict` moves its state to each parameter's device the same way.
  Checkpoints are never saved with `.cpu()` applied first.
- **LOAD-8.** The model's structure always comes from the current `__init__`, called
  through `build_model` with the stored config. The file supplies only the values. A key
  that is missing or extra, or a shape that differs, fails at load time (LOAD-4), not
  later in `forward`.

### 6.5 Translate (RUN, D1)

- **RUN-1.** `translate(saved, text)` takes a `SavedTranslator` and a source sentence
  and returns its greedy translation as a string.
- **RUN-2.** The source row is built by the loader's code, not a copy of it.
  `dataload/multi30k.py` exposes one public function that turns source texts into
  padded source rows (C2), and both the loader and `translate` call it. `L` is
  `config.src_seq_length`.
- **RUN-3.** Decoding follows the training convention (C2, C3). The target prefix
  starts as `[BOS_ID]`. Each step runs `forward` on a `(1, 2, L)` batch with the prefix
  right-padded with `PAD_ID`, and appends the argmax of the logits at the prefix's last
  position. Decoding stops when the argmax is `EOS_ID` or after `L − 1` generated
  tokens, whichever comes first.
- **RUN-4.** `translate` raises `ValueError` when `src_seq_length != tgt_seq_length`,
  because C1 forces one `L` on both rows and the loader trained the model with one
  `seq_len`.
- **RUN-5.** It runs under `torch.no_grad()` in eval mode. The same checkpoint, text,
  and device give the same output on every call.
- **RUN-6.** The output is `bpe.decode` of the generated ids, so protected ids are
  dropped and whitespace comes back as single spaces (C10).
- **RUN-7 (should).** A source longer than `L − 1` tokens is truncated, as in training,
  and `translate` logs a warning when that happens.
- **RUN-8.** `multi30k_translator.py` takes an optional positional command, `train`
  (the default) or `translate`, as the ablation script takes its study name. With no
  command it trains, exactly as today. `translate` prints the translation to stdout
  and logs the language pair from the metadata:

```bash
uv run python app/src/training/model/multi30k_translator.py
uv run python app/src/training/model/multi30k_translator.py translate \
    --checkpoint app/history/checkpoints/multi30k_translator_<run_uuid>.pt \
    --text "A man is riding a bicycle."
```

- **RUN-9.** One sentence per call. Batched decoding is out of scope.

### 6.6 Tests (TEST)

Tests live in `app/tests/test_multi30k_translator.py` (new; the translator has no test
file today), `app/tests/test_bpe.py`, and `app/tests/test_train_utils.py` (new;
`train/utils.py` has none, and the name avoids a clash with the `utils` package). They
need no network and no `app/datasets/`. They use a toy tokenizer such as
`bpe.train(text="low lower lowest", num_merges=5)` and a tiny config: `d_model=16`, one
block a side, two heads, `L=8`.

- **TEST-1.** The config survives JSON:
  `Multi30kTranslatorConfig(**json.loads(json.dumps(asdict(config)))) == config`.
- **TEST-2.** A seeded tiny model, saved with `save_translator` to `tmp_path` and
  loaded with `load_translator`, gives `torch.equal` logits on a fixed batch in eval
  mode. This also confirms C5 and LOAD-5.
- **TEST-3.** A checkpoint written by `save_state` without metadata makes
  `load_translator` raise `ValueError`, and so does one whose `model_class` names
  another class.
- **TEST-4.** `save_state` without metadata writes only `model` and `optimizer`, plus
  `scheduler` when the state has one.
- **TEST-5.** `tokenizer_from_pairs(pairs=pairs_from_merges(merges=merges))` returns
  the `merges` and `vocab` that `bpe.train` returned.
- **TEST-6.** A stub model whose logits follow a scripted token sequence stands in for
  the translator. With it, `translate` reads the logits at the prefix's last position,
  stops at `EOS_ID`, and stops after `L − 1` tokens when `EOS_ID` never comes.
- **TEST-7.** `translate` raises `ValueError` when `src_seq_length != tgt_seq_length`.
- **TEST-8.** A payload with every CKPT-7 field, written with `torch.save` to `tmp_path`,
  loads under the default `weights_only=True`. After the RNG states are restored, the
  next torch and Python `random` draws equal the draws that followed the save.

### 6.7 Docs (DOC)

Per the "Where a change goes" table in `CLAUDE.md`:

- **DOC-1.** The `multi30k_translator.py` module docstring says that `main` trains and
  saves a self-contained checkpoint, and shows the `translate` command.
- **DOC-2.** README, `### Save and restore a checkpoint`: the `metadata` argument, that
  `train/` stores it and never reads it, and when to pass it. The section stays
  general and names no model.
- **DOC-3.** README, "Run something" block: one line for the `translate` command.
- **DOC-4.** README, Project structure: the two new test files, and the translator's
  comment updated to cover translating.
- **DOC-5.** The `bpe.py` docstring (TOK-3) and the `save_state` docstring (CKPT-2).
- **DOC-6.** No change to `pages/overrides/home.html`. Its card says checkpointing
  "stays a named function", which still holds, and the one-screen run does not change
  because `fit` does not (D6). This gets checked again at PR time.

______________________________________________________________________

## 7. Acceptance criteria

The change is done when all of these hold:

1. TEST-1 to TEST-7 pass, and so does the rest of `make test`.
1. `make precheck`, `make check-named-args`, `make check-docstrings`, and
   `sh .githooks/pre-commit` pass.
1. A manual run, with `NUM_EPOCHS` set to 1 locally and not committed, writes
   `app/history/checkpoints/multi30k_translator_<run_uuid>.pt` next to
   `app/history/<run_uuid>.json`. The `translate` command on that checkpoint prints a
   string. One epoch does not make a good translator; this checks the command line.
   The run needs network access, because `main()` passes `cache=False` and downloads
   the data again.
1. `translate` on `multi30k_translator_best.pt` fails with LOAD-4's message.
1. `git diff app/src/training/experiments/` shows only the `NORMS` import.

______________________________________________________________________

## 8. Implementation order

Each step leaves `make test` green.

1. TOK-1 to TOK-3, with TEST-5.
1. CFG-6, the `NORMS` move.
1. CFG-1 to CFG-5, with `main()` building through `build_model`, and TEST-1.
1. CKPT-1 to CKPT-3, with TEST-4.
1. CKPT-4 to CKPT-6 and LOAD-1 to LOAD-6, with TEST-2 and TEST-3.
1. RUN-1 to RUN-9, with TEST-6 and TEST-7.
1. DOC-1 to DOC-6.
1. The manual run in §7.

______________________________________________________________________

## 9. Known limitations and risks

- **The saved weights are the last epoch's.** `main()` trains for 100 epochs. The
  ablation module records that its 100-epoch control run reached its lowest val loss
  at epoch 9 and climbed after
  ([multi30k_translator_ablation.py:138](app/src/training/experiments/multi30k_translator_ablation.py#L138)).
  That run differs from `DEFAULT_CONFIG` (128 wide, dropout 0.1, constant LR), so this
  is a warning, not a measurement: expect `main()`'s checkpoint to be past its best
  epoch. Saving the best epoch is a loop change.
- **Checkpoint size.** The payload carries Adam's two moment buffers per parameter,
  so the file is roughly three times the size of the weights. Running needs none of
  the optimizer state; a weights-only save is a possible follow-up.
- **Two descriptions of the translator,** the ablation `Config` and the new one, until
  the ablation code moves over.
- **Decoding cost.** Each step reruns the encoder and the decoder over the full row, so
  one sentence costs up to `L − 1` forward passes. At `L = 32` that is fine for one
  sentence from the command line.
- **Code drift.** The checkpoint stores the arguments, not the class's code. A later
  change to `Multi30kTranslator` that keeps the same parameter names and shapes loads
  old weights without an error and computes something different. The history file's
  `git_commit` for the same `run_uuid` is the only record of which code produced the
  weights.

______________________________________________________________________

## 10. Open points for review

- LOAD-1 returns a `SavedTranslator` dataclass rather than a 7-tuple like the loader's.
  Confirm the shape.
- RUN-7 is the only "should". Drop it if the warning isn't wanted.
- Whether `SEQ_LEN` should become a single config field rather than a constant that
  fills two (ties to D7).
- The checkpoint filename pattern in CKPT-6.
- How `fit` writes a checkpoint between epochs. ADR 0001 decides that it does and
  leaves the mechanism open. The answer settles CKPT-6 (one save at the end of `main()`)
  and D6, and it reaches the README, `home.html`, and the trainer-design article through
  the "When the loop changes" list in `CLAUDE.md`.
- Which package captures and restores the generic fields (epoch, seeds, RNG states). If
  `train/` owns them, G4 and CKPT-2 narrow to "`train/` interprets only `metadata`
  opaquely", and CKPT-3 ("`load` is unchanged") no longer holds. If the model module
  owns them, each model module repeats that code.
- Where `config` sits: inside `metadata`, as drawn in §5, or at the top level beside
  `model`.
- Whether the tokenizers go in the checkpoint at all, and in what form. See
  [the tokenizer discussion](docs/discussions/bpe-tokenizer-in-checkpoint.md).
- Whether to store the git commit in the checkpoint. The history file already records it
  under the same `run_uuid`.
