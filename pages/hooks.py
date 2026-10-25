"""MkDocs hooks that build the GitHub Pages site from markdown spread across the repo.

Every page keeps its repo path as its src_uri (tutorials/..., README.md),
so relative links between pages resolve exactly as they do on github.com. Links to
anything that isn't a site page (source files, the Makefile, CLAUDE.md) are pointed
at the file on GitHub instead of 404ing.
"""

import posixpath
import re
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

REPO_ROOT = Path(__file__).resolve().parent.parent  # noqa: NAR001
BRANCH = "main"
HOME_PAGE = "README.md"
REPRODUCIBILITY_PAGE = "knowledge_sharing/reproducibility.md"
TOKENISATION_REPORT = "reports/multi30k_tokenisation.md"
ABLATION_REPORT = "reports/multi30k_ablations.md"
STUDY_DETAILS = {
    "experiments/multi30ken_predictor/2026-09-29/README.md": (
        "Masked-token pretraining",
        "34 completed runs",
        "20–50 epochs",
        "Sweep and ablation",
    ),
    "experiments/multi30k_translator/2026-10-13/README.md": (
        "Translator components",
        "30 comparison runs",
        "3 seeds per arm",
        "30 epochs",
    ),
    "experiments/multi30k_translator/2026-10-15/README.md": (
        "Translator scale",
        "27 comparison runs",
        "3 seeds per arm",
        "25 epochs",
    ),
    "experiments/multi30ken_predictor/2026-10-15/README.md": (
        "Masked-token convergence",
        "9 comparison runs",
        "3 seeds per arm",
        "100 epochs",
    ),
}
EXTRA_PAGES = [
    HOME_PAGE,
    REPRODUCIBILITY_PAGE,
    TOKENISATION_REPORT,
    ABLATION_REPORT,
    "app/src/training/common/README.md",
    "app/src/training/transformer/README.md",
    "app/src/training/dataload/README.md",
    "experiments/multi30ken_predictor/2026-09-29/README.md",
    "experiments/multi30k_translator/2026-10-13/README.md",
    "experiments/multi30ken_predictor/2026-10-15/README.md",
    "experiments/multi30k_translator/2026-10-15/README.md",
]

# Images and downloads used by EXTRA_PAGES. Without them as site files, the
# rewriter sends each one to GitHub, which serves HTML instead of the asset.
EXTRA_ASSETS = [
    "reports/assets/multi30k_ablations/training_budget.png",
    "reports/assets/multi30k_ablations/training_budget.svg",
    "reports/assets/multi30k_ablations/schedule_dynamics.png",
    "reports/assets/multi30k_ablations/schedule_dynamics.svg",
    "reports/assets/multi30k_ablations/loss_and_accuracy.png",
    "reports/assets/multi30k_ablations/loss_and_accuracy.svg",
    "reports/assets/multi30k_ablations/translator_components.png",
    "reports/assets/multi30k_ablations/translator_components.svg",
    "reports/assets/multi30k_ablations/width_and_schedule.png",
    "reports/assets/multi30k_ablations/width_and_schedule.svg",
    "reports/assets/multi30k_ablations/masked_token_convergence.png",
    "reports/assets/multi30k_ablations/masked_token_convergence.svg",
    "reports/assets/multi30k_ablations/summary.csv",
    "experiments/multi30ken_predictor/2026-09-29/runs.csv",
    "experiments/multi30ken_predictor/2026-09-29/epochs.csv",
    "experiments/multi30k_translator/2026-10-13/runs.csv",
    "experiments/multi30k_translator/2026-10-13/epochs.csv",
    "experiments/multi30ken_predictor/2026-10-15/runs.csv",
    "experiments/multi30ken_predictor/2026-10-15/epochs.csv",
    "experiments/multi30k_translator/2026-10-15/runs.csv",
    "experiments/multi30k_translator/2026-10-15/epochs.csv",
    "reports/assets/multi30k_tokenisation/mean_length.png",
    "reports/assets/multi30k_tokenisation/mean_length.svg",
    "reports/assets/multi30k_tokenisation/tokens_per_word.png",
    "reports/assets/multi30k_tokenisation/tokens_per_word.svg",
    "reports/assets/multi30k_tokenisation/length_tail.png",
    "reports/assets/multi30k_tokenisation/length_tail.svg",
    "reports/assets/multi30k_tokenisation/stats.csv",
    "knowledge_sharing/assets/rng-state.svg",
    "knowledge_sharing/assets/dawnshard-replay.svg",
    "experiments/multi30ken_predictor/2026-09-29/assets/sweep_best_val.png",
    "experiments/multi30ken_predictor/2026-09-29/assets/val_curves.png",
    "experiments/multi30ken_predictor/2026-09-29/assets/follow_up_best_val.png",
    "experiments/multi30k_translator/2026-10-13/assets/val_curves.png",
    "experiments/multi30k_translator/2026-10-13/assets/best_val_by_arm.png",
    "experiments/multi30ken_predictor/2026-10-15/assets/val_curves.png",
    "experiments/multi30ken_predictor/2026-10-15/assets/best_val_by_arm.png",
    "experiments/multi30k_translator/2026-10-15/assets/val_curves.png",
    "experiments/multi30k_translator/2026-10-15/assets/best_val_by_arm.png",
    "experiments/multi30k_translator/2026-10-15/assets/val_perplexity.png",
    "experiments/multi30k_translator/2026-10-15/assets/val_accuracy.png",
]

README_HERO_PREFIX = "# Dawnshard\n\n**A top-down–designed ML infrastructure library.**\n\n"

# Inline markdown link targets that are relative paths: ](path) or ](path#anchor).
RELATIVE_LINK = re.compile(pattern=r"\]\((?![a-z]+:|#)([^)\s#]+)(#[^)\s]*)?\)")


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Re-root docs_dir files under their repo path and add the extra pages and assets.

    Theme assets keep their own src_dir and are left alone.
    """
    docs_dir = Path(config.docs_dir).resolve()  # noqa: NAR001
    docs_prefix = docs_dir.relative_to(REPO_ROOT).as_posix()  # noqa: NAR001
    for file in list(files):  # noqa: NAR001
        if file.src_dir == config.docs_dir:
            files.remove(file=file)
            files.append(file=_repo_file(path=f"{docs_prefix}/{file.src_uri}", config=config))
    for path in EXTRA_PAGES + EXTRA_ASSETS:
        files.append(file=_repo_file(path=path, config=config))
    return files


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Use the landing template for the home page and send non-page links to GitHub."""
    if page.file.src_uri == REPRODUCIBILITY_PAGE:
        page.meta["template"] = "reproducibility.html"
        page.meta["hide"] = ["navigation", "toc"]
    if page.file.src_uri == TOKENISATION_REPORT:
        page.meta["template"] = "report.html"
        page.meta["hide"] = ["navigation", "toc"]
        page.meta["report_subject"] = "Multi30k tokenisation"
        page.meta["description"] = (
            "How BPE vocabulary size changes English and German Multi30k caption "
            "lengths, tokens per word, and the long-input tail."
        )
    if page.file.src_uri == ABLATION_REPORT:
        page.meta["template"] = "report.html"
        page.meta["hide"] = ["navigation", "toc"]
        page.meta["report_subject"] = "Multi30k ablation studies"
        page.meta["report_period"] = "October 13–15, 2026"
        page.meta["description"] = (
            "Learn to read transformer ablations through learning curves: "
            "overfitting, dropout, model size, learning rates, and loss versus accuracy."
        )
    if page.file.src_uri in STUDY_DETAILS:
        subject, *facts = STUDY_DETAILS[page.file.src_uri]
        page.meta["template"] = "experiment.html"
        page.meta["study_subject"] = subject
        page.meta["study_facts"] = facts
        page.meta["study_date"] = page.file.src_uri.split(sep="/")[-2]
    if page.file.src_uri == HOME_PAGE:
        page.meta["template"] = "home.html"
        page.meta["hide"] = ["navigation"]
        if not markdown.startswith(README_HERO_PREFIX):  # noqa: NAR001
            raise ValueError(  # noqa: NAR001
                "README.md title or tagline no longer matches the site hero"
            )
        # The hero supplies these two lines; the template renders the rest below it.
        markdown = markdown[len(README_HERO_PREFIX) :]  # noqa: NAR001
    page_dir = posixpath.dirname(p=page.file.src_uri)

    def rewrite(match: re.Match[str]) -> str:
        """Return the link unchanged if it targets a site page, else a GitHub URL."""
        target, anchor = match.group(1), match.group(2) or ""  # noqa: NAR001
        resolved = posixpath.normpath(path=posixpath.join(page_dir, target))  # noqa: NAR001
        on_disk = REPO_ROOT / resolved
        if resolved.startswith("..") or not on_disk.exists():  # noqa: NAR001
            return match.group(0)  # noqa: NAR001
        if files.get_file_from_path(path=resolved):
            return match.group(0)  # noqa: NAR001
        readme = posixpath.join(resolved, "README.md")  # noqa: NAR001
        if on_disk.is_dir() and files.get_file_from_path(path=readme):
            # A bare directory link only works from pages at the same URL depth;
            # pointing at the README lets MkDocs compute the right relative URL.
            return f"]({posixpath.relpath(path=readme, start=page_dir or '.')}{anchor})"
        kind = "tree" if on_disk.is_dir() else "blob"
        return f"]({config.repo_url}/{kind}/{BRANCH}/{resolved}{anchor})"

    return RELATIVE_LINK.sub(rewrite, markdown)  # noqa: NAR001


def _repo_file(path: str, config: MkDocsConfig) -> File:
    """Build a File whose source is `path` relative to the repo root."""
    return File(
        path=path,
        src_dir=str(REPO_ROOT),  # noqa: NAR001
        dest_dir=config.site_dir,
        use_directory_urls=config.use_directory_urls,
    )
