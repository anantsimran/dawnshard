# CLAUDE.md

## Named-argument linting

All function calls must use keyword arguments. A custom checker (`scripts/check_named_args.py`) enforces this as part of `make precheck`.

**Suppress with `# noqa: NAR001`** when keyword arguments are genuinely unavailable:

- C builtins and C extension methods: `print`, `len`, `set`, `range`, `iter`, `next`, `isinstance`, `sorted`, `zip`, `str`, `type`, `list.append`, `dict.update`, etc.
- `nn.Module.__call__` (PyTorch forward pass): `model(x)`, `criterion(pred, target)`
- `nn.Sequential(*modules)` and other `*args`-based constructors
- `Callable` type aliases where parameter names are not part of the contract
- loguru `logger.info(msg, *args)` and similar format-style logging

**Do not suppress** calls to regular Python functions and methods that have named parameters — use keyword arguments instead.

The ruff linter is configured to allow `NAR001` as an external code via `pyproject.toml`:

```toml
[tool.ruff.lint]
external = ["NAR001"]
```

## Docstring linting

Every function and method needs a Google-style docstring, except in test files. A custom checker (`scripts/check_docstrings.py`) enforces the presence of a docstring — it does not check style or content, just that one exists.

Run it manually with `make check-docstrings`, or install it as a git pre-commit hook (checks all of `app/src` on every commit) with `make install-hooks`.

**Suppress with `# noqa: NAR002`** on the `def` line when a docstring genuinely doesn't apply (e.g. a trivial dunder method).

The ruff linter is configured to allow `NAR002` as an external code via `pyproject.toml`:

```toml
[tool.ruff.lint]
external = ["NAR001", "NAR002"]
```

## Documenting an `nn.Module`

The checker above only asks that a docstring exists. This section says what it has to contain. Every `nn.Module` under `app/src` explains its forward pass, and explains it by drawing it. `MultiHeadAttentionLayer` and `EncoderBlock` in [modules.py](app/src/training/transformer/modules.py) are the reference; `SinusoidalEmbedding` and `AGNewsClassifier` follow them.

Write the class docstring in this order:

1. **A one-line summary that is the pipeline**, arrows and all: `Embedding → 2 pre-norm encoder blocks → mask-aware mean pool → 4 logits.`
1. **A short paragraph** saying what the module is for and what goes in and out, with shapes: "In and out are both (B, L, d_model), so blocks stack by composition."
1. **A fenced ASCII diagram of the forward pass.** Rules below.
1. **One paragraph per non-obvious choice**, each saying what that choice buys. Pre-norm over post-norm, one shared `nn.Dropout`, a mean pool instead of a `[CLS]`, a non-persistent buffer. If a reader could mistake the choice for a bug, this is where it stops being one.
1. **A closing line about output or mutable state** — raw logits vs. probabilities, or which attribute `forward` overwrites.

### Diagram rules

- Box-drawing characters only: `│ ─ ▼ ► ┌ ┐ └ ┘ ├ ┴ ┬`. No ASCII art with `|` and `-`, no emoji.
- Data flows top to bottom. Annotate each wire with the shape it carries (`│  (B, L, d_model)`), and put commentary in a right-hand column (`padding_idx=PAD_ID`, `RMSNorm over d_k, only if qk_norm`).
- Name the actual attributes and functions the diagram stands for — `embedding_dropout`, `pad_masked_mean`, `@ w_o` — so the diagram can be grepped against the code.
- Shapes use the axis vocabulary in [transformer/constants.py](app/src/training/transformer/constants.py): B, L, d_model, h, d_k. A new axis goes in that table first.
- Fence it with triple backticks inside the docstring, and keep every line inside the 88-character limit. `ruff format` does not reflow docstrings, so this is on you — and count **characters**, not bytes: `awk 'length > 88'` reports every box-drawing character as three and will lie to you.

### The methods

- `__init__`: what it builds, then `Args:` for the parameters whose meaning isn't obvious from the name, then `Raises:` for every validation it performs. Don't restate `d_model: The model dimension.`
- `forward`: one line on what it returns, `Args:` covering what the caller must guarantee (padding convention, what a mask's `True` means, which builder produced it), and `Returns:` when the return is a tuple.
- **Any method that mutates `self` gets a `Side effects:` section.** `AGNewsClassifier.forward` overwrites `attention_map` on every call, and a probe reads it; that fact lives in the docstring, not in a comment.

### What does not go in a module docstring

Package-level narrative — how attention is built, why masks are boolean — lives in the package README (`app/src/training/transformer/README.md`), and the module docstring points at it rather than repeating it. Dataset choices live in the dataload module's own docstring. See Architecture choices below for which file owns which decision.

## Keep general code and docs general

This is a general-purpose library. Code under `common/`, `transformer/`, `train/`, `metrics/`, `viz/`, and `utils/` works for any dataset and any model, so its comments, docstrings, and package READMEs must not name one: no AG News, MNIST, Multi30k, or a model class from `model/`. Examples use a generic `model`, `loader`, `src_text`, or a toy string such as `bpe.train(text="low lower lowest", ...)`. Say what a caller must pass, not which caller does ("`vocab`: as returned by `bpe.train`", not "as returned by `get_ag_news_dataloader`").

A dataset or model belongs in exactly three places: its own module (`dataload/<name>.py`, `model/<name>.py`, and a script that exists for one dataset, such as `viz/bpe_seq_len.py`), the architecture-choices list below, and the root README sections that inventory the repo (What's inside, Getting started, Project structure). The package READMEs, including `dataload/README.md`, stay general.

## Line length

The project enforces an 88-character line limit (configured in `pyproject.toml`). `ruff format` (run by `make precheck`) handles most wrapping automatically, but **cannot** split lines that contain `# noqa` comments or long string literals — those must be wrapped manually.

The standard pattern is to open a parenthesis and move the `# noqa` to that first line:

```python
# Before (too long):
some_call(arg1, arg2, arg3)  # noqa: NAR001

# After:
some_call(  # noqa: NAR001
    arg1,
    arg2,
    arg3,
)
```

## Pre-commit checks

These checks have standing permission. Run them without asking, before reporting a change as done:

```bash
make precheck           # pyright, ruff check --fix, ruff format, mdformat
make check-named-args
make check-docstrings
make test               # or: make test DIR=app/tests/test_x.py
sh .githooks/pre-commit
```

`make precheck` rewrites files (ruff `--fix`, `ruff format`, `mdformat .`), so review its diff before reporting. To tell whether a failure is pre-existing, compare against `git show HEAD:<path>`. Don't `git stash` the working tree, since the user may be editing files at the same time.

## Before creating a PR

Where the root `README.md` or the site pages (`tutorials/`, `tutorials/index.md`, `mkdocs.yml`, `pages/overrides/home.html`, the READMEs in `EXTRA_PAGES`) no longer match, update them in the same PR, using Where a change goes under README structure to pick the sections.

## Tutorials site

`tutorials/` and the root `README.md` are published to GitHub Pages with MkDocs Material (`mkdocs.yml`, `pages/`, `.github/workflows/pages.yml`).

Tutorials are grouped by topic: `pytorch/` (basics), `training/` (loop, optimizers, schedulers, observability), `math/`, `vision/`, and `transformer/`. Put a new page in the folder that matches its topic.

**Whenever you add, remove, move, or change a tutorial, update `tutorials/index.md` in the same change.** The index has two parts, and both must stay accurate:

- **Pages:** one row per page, in its folder's table, with a one-line summary of what it covers.
- **Concepts:** question → `page.md#section` rows. Add rows for new sections, and fix or remove rows whose heading you renamed or deleted, because the anchor comes from the heading text. The mkdocs build below warns about anchors that no longer exist.

**When you add a tutorial, add a line for it to the `nav:` list in `mkdocs.yml`.** Otherwise the page is built but doesn't appear in the site's navigation. The build prints `WARNING - The following pages exist in the docs directory, but are not included in the "nav" configuration`, followed by the missing page.

To publish another page from outside `tutorials/`, add its path to `EXTRA_PAGES` in `pages/hooks.py` and add a `nav:` entry for it. If it embeds images, add each one to `EXTRA_ASSETS` in the same file, or the link rewriter points them at GitHub and they render as a link instead of an image.

**The landing page is not generated from the README.** `pages/overrides/home.html` is a hand-written hero and card grid that renders above the README and copies parts of it: the hero code block is the README's one-screen run, the "Design choices" cards summarize the design sections, and the "Learning track" cards list what each tutorial group covers. Nothing checks it. The mkdocs build passes and the pre-PR hook is satisfied while it goes stale. So in the same change:

- When the README's one-screen run changes, copy it into the hero `<pre>` block with the same argument names.
- When a claim a design card makes changes in the README (what `EpochSpec` holds, what the history file records, what the loop injects, a rule or tool), rewrite that card.
- When a tutorial is added, moved, or regrouped, fix the learning-track card for its group.

Before opening the PR, read `home.html` next to the README diff. Don't treat a green build as proof the landing page is current.

Check a change with:

```bash
uvx --from mkdocs==1.6.1 --with mkdocs-material==9.7.7 --with pymdown-extensions==12.0 mkdocs build
```

Keep these versions in step with the pins in `pages.yml`. MkDocs 2.0 removes the hook and theme-override system the site depends on.

## README structure

The root `README.md` is the repo's front page and the site's landing page. It is written for someone evaluating the repo as a piece of engineering, not for someone who already uses it. It has to sell the design before it explains it: the reader sees a real training run, then the principle behind it, then the rules the repo enforces, then the reasoning, and only then the setup. Keep that order. Don't move practical material (install, commands, Docker) above the design sections, and don't move the design reasoning below them.

The voice is an engineer showing their work, slightly formal. No first person: never "I", "my", or "we", with one deliberate exception, the opening line of the principle section ("the part I am proudest of"), which stays. Stay in the active voice anyway by making the code the subject ("`fit` writes", "the hook refuses to open the PR") or addressing the reader ("you"). Short sentences, concrete claims that point at a file. Never marketing filler and never the flat, hedged register of generated text.

### The opening

The file starts with exactly this shape, and `pages/hooks.py` depends on it:

1. `# Dawnshard` on the first line.
1. A bold one-line tagline (`**A top-down–designed ML infrastructure library.**`).
1. The two intro paragraphs: "Dawnshard is an ML stack built from first principles…" and "Most ML code grows bottom-up…". **These are user-approved and stay verbatim.** Don't paraphrase, tighten, or "fix" them.
1. The long-term goal paragraph (Constitutional AI and RLHF).
1. `## Why the name`, the Cosmere story, as written. It stays here, directly after the intro, and it stays verbatim.

`pages/hooks.py` strips the title and the bold tagline because the landing-page hero already shows them. A second bold line, a badge row, or a table of contents before the intro would either be stripped by mistake or render twice.

### Section order

Top-level sections are `##`, separated by a horizontal rule. They are not numbered, so each anchor comes from the heading text alone (`## What's inside` → `#whats-inside`). This is the order, and what each section is for:

| Section | Purpose | Rules |
|---|---|---|
| `## A training run in one screen` | Show the design before explaining it. One training run of `TrainState`, `EpochSpec`, and `fit` over a generic `model`, `train_loader`, and `val_loader`, then a three-bullet list naming the three objects and their jobs, and one line saying that is the entire API. | **The library is general, so this section must be too.** The snippet names no dataset and no model class. Keep the code under 25 lines and keep it a real call sequence, not pseudocode. No explanation beyond the three bullets; the next section does that. If the loop's API changes, this block changes in the same PR. |
| `## Separate state, policy, and mechanism` | **The most strongly advertised idea on the page.** Names the principle (separation of policy and mechanism, from Hydra, 1974, plus explicit state as in JAX/Flax, whose struct is also called `TrainState`) and maps every component to exactly one of the three. | Four `###` subsections, in this order: `State: TrainState` (the only object the loop mutates; checkpoints are its `state_dict`s), `Policy: EpochSpec` (read by the loop, never written), `Mechanism: the functions` (shows the `StepFn` type alias from `train_loop.py` and reads it as "state in, policy in, data in, metrics out"), and `What that buys you` (bold-lead bullets: one loop for train and eval, tests that pass a plain list as the loader, no `self` to grep, every piece replaceable alone). Every claim points at the file that proves it. If a component is added to the loop, add it to one of the three subsections; it must fit exactly one. |
| `## Exactly one place for everything` | The consequence of the principle: every extension has one known home, so there is no hidden complexity. | Four `###` subsections: `Metrics: a function of the outputs`, `Probes: a function of the training state`, `Everything else: a named function` (a question-and-answer bullet per function: `profiled_fit`, `save_state`/`load`, the history file), and `No hidden complexity`. Keep it about where things go, not how they are implemented. No tooling or lint rules here; those belong in the next section. |
| `## The repo keeps itself clean` | The engineering pitch: how the layout reinforces hygiene, then every rule the repo enforces and the tool that enforces it. | Two `###` subsections. `The layout enforces the boundaries`: a `Package | Owns` table with one row per package under `app/src/training/`, then a bullet list of the rules that fall out of the layout (downward-only imports, one `constants.py` and one `setup.py`, gitignored artifacts, one test file per module). `Every rule has a tool behind it`: a `Rule | Enforced by` table, one row per rule, each rule phrased "Every …" and each enforcer naming the file. A rule only belongs here if a tool checks it. When you add such a tool, add a row. When you remove one, remove its row. When you add a package, add a row to the layout table. |
| `## Debuggability` | What a run leaves behind and how to read it. | `###` subsections in this order: `The terminal: one line per epoch`; `The history file: everything` (a numbered list of the three functions that build it: `EpochRecord` → `serialize_epoch_record` → `save_history`, then a trimmed real excerpt, then a bold-lead bullet list of what the file reveals about that run and a one-line verdict); `Plot it locally` (the `plot_metrics.py` command and its screenshot); `Compare two runs`; `Weights & Biases: the same dict, streamed`; `When the numbers aren't enough` (pointers to probes and `profiled_fit`). The excerpt and the screenshot come from a real run in `app/history/`; regenerate both when the history format or the plot script changes. Keep the excerpt trimmed to one epoch and rounded. The screenshot lives in `tutorials/assets/` (see Style). |
| `## What's inside` | A scannable map: one table row per part of the library, with a one-line description and a link to its package. | Add a row when you add a package or a major capability. Keep descriptions to one line. |
| `## Using the loop` | The manual: how each part of the loop behaves and how to use it. | One `###` per task, in this order: `Train and validate`, `Seed a run`, `Repeat a past run`, `Add a metric`, `Add a probe`, `Profile a few epochs`, `Save and restore a checkpoint`. Each opens with a snippet, then says what the loop does with it and, where it isn't obvious, why. Only mechanics and reasoning that no earlier section states belong here. Don't re-explain the state/policy/mechanism split, the history file (Debuggability owns it), or where things go (Exactly one place for everything owns it); link to them instead. `Train and validate` shows only the imports and points at the one-screen run rather than repeating it. A new loop feature gets one `###`. |
| `## The BPE tokenizer` | A short section: what it is, one snippet, one number. | **Keep it short. BPE internals stay out of the README**; they live in `bpe.py`'s docstrings and in the intentional-choices list below. |
| `## Getting started` | Everything practical, as `###` subsections: install, run something, W&B, the MNIST models, model graph, attention heatmaps, Docker, Make targets. | A new script gets a line in the "Run something" block. A new Make target gets a row in the Make targets table. |
| `## Tutorials` | What the learning track covers and how the site is built. | Keep the bold reminder about `tutorials/index.md` and `mkdocs.yml`. |
| `## Project structure` | The annotated tree. It closes the page. | Add every new file or folder that a reader would look for. One-line comments, aligned with the rest. |

### The transformer README

Attention design lives in `app/src/training/transformer/README.md`, not the root README. The root README's What's inside table links to it. It has two sections: `## Attention` (the reasoning behind `attention`, `MultiHeadAttentionLayer`, `EncoderBlock`, and the masks, explaining each choice by what it buys, plus a usage snippet) and `## Axis names` (the symbol table). Any new axis name goes in that table and in `transformer/constants.py` in the same change. `constants.py` and `modules.py` point readers at this README in their docstrings. It is published on the site through `EXTRA_PAGES` in `pages/hooks.py` and the Transformers `nav:` group in `mkdocs.yml`. Links to package files are relative to the package (`[constants.py](constants.py)`).

### The dataload README

How a `DataLoader` is put together lives in `app/src/training/dataload/README.md`: the sampler → dataset → `collate_fn` pipeline, what a new loader has to define, shuffling, train and eval loaders, seeding, chunking long documents, and a checklist for writing a new loader. It is general, per Keep general code and docs general above: it names no dataset, and a dataset's own choices go in its module docstring. The basics (why mini-batches, transforms, normalization) stay in `tutorials/pytorch/dataloading.md`, and the README links there instead of repeating them. The root README's What's inside table links to it. It is published through `EXTRA_PAGES` in `pages/hooks.py` and the PyTorch `nav:` group in `mkdocs.yml`.

### Anchors other files depend on

`pages/overrides/home.html` links to `#a-training-run-in-one-screen`, `#separate-state-policy-and-mechanism`, `#exactly-one-place-for-everything`, `#the-repo-keeps-itself-clean`, and `#debuggability`, and its design cards summarize the README. Those links only resolve because `pages/hooks.py` renders the README below the hero, so they are anchors on the landing page itself. The walkthrough links to `#visualizing-attention` and `#debuggability`, and Debuggability links to `#add-a-probe` and `#profile-a-few-epochs`. When you rename or remove one of those headings, update the links in the same change, then run the mkdocs build above; it warns about anchors that no longer exist.

### Where a change goes

Use this to decide which README sections a code change touches. A pre-PR hook (`.claude/hooks/check_tests_before_pr.py`) blocks the PR if `app/src/` changed and `README.md` didn't, so the answer is rarely "nowhere".

| You changed | Update |
|---|---|
| The loop's API (`fit`, `EpochSpec`, `TrainState`, probes) | The one-screen run, the matching `###` under Separate state, policy, and mechanism, the matching `###` under Exactly one place for everything, and the matching `###` under Using the loop. |
| A rule or the tool that enforces it (linter, hook, Make target, typing) | Its paragraph under The repo keeps itself clean, and the Make targets table if a target changed. |
| A package, module, or major capability | Its row in What's inside, its line in Project structure, and the layout paragraph under The repo keeps itself clean if it is a new package. |
| A script that is run from the command line | The "Run something" block. |
| The history file format, `serialize_epoch_record`, `save_history`, or a viz script | Debuggability: the prose, the JSON excerpt, and the screenshot (regenerate it with the `plot_metrics.py` command shown there, then extract the PNG from `/tmp/plot_metrics.html` into `tutorials/assets/plot_metrics.png`). |
| An attention or mask choice | `app/src/training/transformer/README.md` (not the root README), and the intentional-choices list below if it's a choice that looks like a bug. |
| How loaders are built: collate, samplers, shuffling, seeding, workers | `app/src/training/dataload/README.md`, kept general. |
| A BPE or AG News data choice | The docstring in `bpe.py` or `ag_news_classifier.py`, and the intentional-choices list below. Not the README. |
| A tutorial | `tutorials/index.md` and `mkdocs.yml`, per the Tutorials site section above, and the learning-track card in `pages/overrides/home.html` if the group's topics changed. |
| Any README section the landing page summarizes (the one-screen run, the design sections, debuggability, the rules) | The hero code block and the matching design card in `pages/overrides/home.html`. See the Tutorials site section above. |

### Style

- Plain, specific prose in the active voice. Say what a thing does and what it buys, in that order. No marketing filler ("seamless", "powerful", "robust"), no emoji. Questions are fine when the next sentence is the one answer.
- Structure every top-level section so it can be skimmed. Break it into `###` subsections with a noun-phrase title (`State: TrainState`, `Plot it locally`). Keep paragraphs to three or four sentences. Use a bullet list for parallel items, with a bold lead-in when each bullet makes a distinct claim (`- **It overfits.** …`). Use a table when the section is a mapping (`Rule | Enforced by`, `Package | Owns`). A wall of prose is a bug.
- Code before explanation. Every walkthrough step opens with a runnable snippet, and every snippet uses keyword arguments, as the repo does.
- Link to source with relative paths (`[train_loop.py](app/src/training/train/train_loop.py)`). The site hook rewrites these to GitHub, so they work in both places.
- Images the README embeds go in `tutorials/assets/`, referenced as `![alt](tutorials/assets/name.png)`. That folder is inside the site's `docs_dir`, so the hook serves the file instead of rewriting the link to a GitHub page, which would not render as an image. Anywhere else in the repo breaks on the site. Write a real alt text that says what the plot shows.
- `make precheck` runs `mdformat` over the file. Write in its style so it makes no changes: horizontal rules are a line of underscores, every ordered-list item is `1.`, and tables use `|---|` separators. Run `uv run mdformat README.md` before committing.

## Architecture choices

The AG News data notes live in the module docstring of `app/src/training/model/ag_news_classifier.py`, not the README. The attention notes live in `app/src/training/transformer/README.md`. **When a change alters one of the choices below, update its README section (or that docstring) in the same change.**

These choices are intentional, even where they look like bugs. Don't "fix" them without asking:

- **Training loop:** there is no `Trainer` class. State (`TrainState`), policy (`EpochSpec`), and functions (`fit`, `run_epoch`, `train_step`, `eval_step`) stay separate. Checkpointing and history are functions you call, not lifecycle hooks.
- **Attention:** `attention` works on pre-split heads and returns `(out, weights)`. Masks are boolean, and `True` means keep. Padding masks hide key columns only, never whole query rows, because a fully masked row turns into NaN after softmax.
- **BPE (`app/src/training/common/bpe.py`):** all whitespace becomes a single space, so `decode` is lossy. Text is pre-tokenized with the GPT-4 (`cl100k_base`) pattern and merges never cross a chunk; a leading space or symbol joins the next word, and numbers split into runs of at most 3 digits. Overlapping runs merge left to right (`aaa` → `aa` + `a`). The protected ids (`PAD_ID = 256`, then `MASK_ID`, `CLS_ID`, `SEP_ID`) live in `common/constants.py` and sit between the byte ids and the merge ids, so they don't move when `num_merges` changes. `FIRST_MERGE_ID` is derived in `bpe.py` as `256 + len(PROTECTED_IDS)`.
- **AG News (`app/src/training/dataload/ag_news.py`):** the tokenizer trains on train rows only. The row cache only checks that files exist, so callers pass `refresh_cache=True` after changing `n_*` or `seed`. The DataLoaders use no worker processes, because the data is already tokenized in memory.
