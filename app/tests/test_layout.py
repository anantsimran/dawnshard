import ast
from graphlib import TopologicalSorter
from pathlib import Path

TRAINING_ROOT = Path(__file__).parent.parent / "src" / "training"

# The packages each package under app/src/training may import. Adding an edge is a
# design decision, and this table is where it gets made. Imports under
# `if TYPE_CHECKING:` count: needing another package's types is depending on it.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "constants": set(),  # noqa: NAR001
    "setup": set(),  # noqa: NAR001
    "common": set(),  # noqa: NAR001
    "transformer": set(),  # noqa: NAR001
    "utils": set(),  # noqa: NAR001
    "train": {"constants", "utils"},
    "metrics": {"constants", "train"},
    "dataload": {"common", "constants", "utils"},
    "viz": {"common", "constants", "dataload", "transformer"},
    "analytics": {"common", "constants", "dataload"},
    "model": {
        "common",
        "constants",
        "dataload",
        "metrics",
        "setup",
        "train",
        "transformer",
        "viz",
    },
}


def _package_of(path: Path) -> str:
    relative = path.relative_to(TRAINING_ROOT)  # noqa: NAR001
    return relative.parts[0].removesuffix(".py")  # noqa: NAR001


def _imported_packages(path: Path) -> set[str]:
    modules = []
    for node in ast.walk(node=ast.parse(source=path.read_text())):
        if isinstance(node, ast.Import):  # noqa: NAR001
            modules.extend(alias.name for alias in node.names)  # noqa: NAR001
        elif isinstance(node, ast.ImportFrom) and node.module:  # noqa: NAR001
            modules.extend(  # noqa: NAR001
                f"{node.module}.{alias.name}" for alias in node.names
            )
    return {
        module.split(sep=".")[1]
        for module in modules
        if module.startswith("training.")  # noqa: NAR001
    }


def test_allowed_imports_have_no_cycle():
    # static_order raises CycleError if any package can reach itself.
    list(TopologicalSorter(graph=ALLOWED_IMPORTS).static_order())  # noqa: NAR001


def test_every_package_is_in_the_table():
    packages = {_package_of(path=path) for path in TRAINING_ROOT.rglob(pattern="*.py")}
    assert packages == ALLOWED_IMPORTS.keys()


def test_imports_follow_the_table():
    violations = []
    for path in sorted(TRAINING_ROOT.rglob(pattern="*.py")):  # noqa: NAR001
        package = _package_of(path=path)
        allowed = ALLOWED_IMPORTS[package] | {package}
        relative = path.relative_to(TRAINING_ROOT)  # noqa: NAR001
        for imported in sorted(_imported_packages(path=path) - allowed):  # noqa: NAR001
            violations.append(f"{relative} imports training.{imported}")  # noqa: NAR001
    assert violations == []
