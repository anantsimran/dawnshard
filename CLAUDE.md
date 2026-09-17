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

## Tutorials site

`tutorials/` and the root `README.md` are published to GitHub Pages with MkDocs Material (`mkdocs.yml`, `pages/`, `.github/workflows/pages.yml`).

Tutorials are grouped by topic: `pytorch/` (basics), `training/` (loop, optimizers, schedulers, observability), `math/`, `vision/`, and `transformer/`. Put a new page in the folder that matches its topic.

**Whenever you add, remove, move, or change a tutorial, update `tutorials/index.md` in the same change.** The index has two parts, and both must stay accurate:

- **Pages:** one row per page, in its folder's table, with a one-line summary of what it covers.
- **Concepts:** question → `page.md#section` rows. Add rows for new sections, and fix or remove rows whose heading you renamed or deleted, because the anchor comes from the heading text. The mkdocs build below warns about anchors that no longer exist.

**When you add a tutorial, add a line for it to the `nav:` list in `mkdocs.yml`.** Otherwise the page is built but doesn't appear in the site's navigation. The build prints `WARNING - The following pages exist in the docs directory, but are not included in the "nav" configuration`, followed by the missing page.

To publish another README, add its path to `EXTRA_PAGES` in `pages/hooks.py` and add a `nav:` entry for it.

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
| `## Why the training loop is different` | The reasoning behind the loop's seams, then the usage walkthrough. | `### Design decisions` holds one `####` per decision (one loop for train and eval, injected metrics, probes, history, profiling, checkpoints). `### Using the loop` holds one `####` per task (train, read history, add a metric, add a probe, profile, checkpoint), each with a runnable snippet. A new loop feature gets both a decision and a walkthrough step. |
| `## The BPE tokenizer` | A short section: what it is, one snippet, one number. | **Keep it short. BPE internals stay out of the README**; they live in `bpe.py`'s docstrings and in the intentional-choices list below. |
| `## Getting started` | Everything practical, as `###` subsections: install, run something, W&B, the MNIST models, model graph, attention heatmaps, Docker, Make targets. | A new script gets a line in the "Run something" block. A new Make target gets a row in the Make targets table. |
| `## Tutorials` | What the learning track covers and how the site is built. | Keep the bold reminder about `tutorials/index.md` and `mkdocs.yml`. |
| `## Project structure` | The annotated tree. It closes the page. | Add every new file or folder that a reader would look for. One-line comments, aligned with the rest. |

### The transformer README

Attention design lives in `app/src/training/transformer/README.md`, not the root README. The root README's What's inside table links to it. It has two sections: `## Attention` (the reasoning behind `attention`, `MultiHeadAttentionLayer`, `EncoderBlock`, and the masks, explaining each choice by what it buys, plus a usage snippet) and `## Axis names` (the symbol table). Any new axis name goes in that table and in `transformer/constants.py` in the same change. `constants.py` and `modules.py` point readers at this README in their docstrings. It is published on the site through `EXTRA_PAGES` in `pages/hooks.py` and the Transformers `nav:` group in `mkdocs.yml`. Links to package files are relative to the package (`[constants.py](constants.py)`).

### Anchors other files depend on

`pages/overrides/home.html` links to `#separate-state-policy-and-mechanism`, `#whats-inside`, `#why-the-training-loop-is-different`, `#the-repo-keeps-itself-clean`, and `#debuggability`, and its design cards summarize the README. The walkthrough links to `#visualizing-attention` and `#debuggability`, and Debuggability links to `#probes-look-inside-the-model` and `#profiling-is-a-separate-function`. When you rename or remove one of those headings, update the links in the same change, then run the mkdocs build above; it warns about anchors that no longer exist.

### Where a change goes

Use this to decide which README sections a code change touches. A pre-PR hook (`.claude/hooks/check_tests_before_pr.py`) blocks the PR if `app/src/` changed and `README.md` didn't, so the answer is rarely "nowhere".

| You changed | Update |
|---|---|
| The loop's API (`fit`, `EpochSpec`, `TrainState`, probes) | The one-screen run, the matching `###` under Separate state, policy, and mechanism, the matching `###` under Exactly one place for everything, the matching `####` under Design decisions, and the matching `####` under Using the loop. |
| A rule or the tool that enforces it (linter, hook, Make target, typing) | Its paragraph under The repo keeps itself clean, and the Make targets table if a target changed. |
| A package, module, or major capability | Its row in What's inside, its line in Project structure, and the layout paragraph under The repo keeps itself clean if it is a new package. |
| A script that is run from the command line | The "Run something" block. |
| The history file format, `serialize_epoch_record`, `save_history`, or a viz script | Debuggability: the prose, the JSON excerpt, and the screenshot (regenerate it with the `plot_metrics.py` command shown there, then extract the PNG from `/tmp/plot_metrics.html` into `tutorials/assets/plot_metrics.png`). |
| An attention or mask choice | `app/src/training/transformer/README.md` (not the root README), and the intentional-choices list below if it's a choice that looks like a bug. |
| A BPE or AG News data choice | The docstring in `bpe.py` or `ag_news_classifier.py`, and the intentional-choices list below. Not the README. |
| A tutorial | `tutorials/index.md` and `mkdocs.yml`, per the Tutorials site section above. |

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
- **BPE (`app/src/training/common/bpe.py`):** all whitespace becomes a single space, so `decode` is lossy. Merges may cross word boundaries. Overlapping runs merge left to right (`aaa` → `aa` + `a`). `PAD_ID = 256` sits between the byte ids and the merge ids, so it doesn't move when `num_merges` changes.
- **AG News (`app/src/training/dataload/ag_news.py`):** the tokenizer trains on train rows only. The row cache only checks that files exist, so callers pass `refresh_cache=True` after changing `n_*` or `seed`. The DataLoaders use no worker processes, because the data is already tokenized in memory.
