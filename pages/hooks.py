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
EXTRA_PAGES = [
    HOME_PAGE,
    "app/src/training/transformer/README.md",
    "app/src/training/dataload/README.md",
]
EXTRA_ASSETS = ["graphify-out/graph.html"]

# Inline markdown link targets that are relative paths: ](path) or ](path#anchor).
RELATIVE_LINK = re.compile(pattern=r"\]\((?![a-z]+:|#)([^)\s#]+)(#[^)\s]*)?\)")
# The README's title and bold tagline, which the landing page hero already shows.
HOME_TITLE = re.compile(pattern=r"\A# .*\n+(\*\*.*\*\*\n+)?")


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Re-root docs_dir files under their repo path and add the READMEs as pages.

    Theme assets keep their own src_dir and are left alone.
    """
    docs_dir = Path(config.docs_dir).resolve()  # noqa: NAR001
    docs_prefix = docs_dir.relative_to(REPO_ROOT).as_posix()  # noqa: NAR001
    for file in list(files):  # noqa: NAR001
        if file.src_dir == config.docs_dir:
            files.remove(file=file)
            files.append(file=_repo_file(path=f"{docs_prefix}/{file.src_uri}", config=config))
    for path in EXTRA_PAGES:
        files.append(file=_repo_file(path=path, config=config))
    for path in EXTRA_ASSETS:
        files.append(file=_repo_file(path=path, config=config))
    return files


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Use the landing template for the home page and send non-page links to GitHub."""
    if page.file.src_uri == HOME_PAGE:
        page.meta["template"] = "home.html"
        page.meta["hide"] = ["navigation"]
        markdown = HOME_TITLE.sub("", markdown, count=1)  # noqa: NAR001
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
