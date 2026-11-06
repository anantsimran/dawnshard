# Dal

This package owns storing the results of expensive steps. `@cached` stores a
function's return value in a `Dao`, keyed by the function and its arguments and tagged
with a version, and `SqliteDao` keeps those entries in one SQLite file. A caller opts
in by passing a `Dao`; with none, every function computes as if the package did not
exist.

| Owns | Imports from the library | Never sees |
|---|---|---|
| the `Dao` protocol, its SQLite implementation, `@cached` | nothing | datasets, tokenizers, models, the training loop |

[test_layout.py](../../../tests/test_layout.py) fails if this package imports anything
from `training/`, so any package can depend on it without a cycle.

______________________________________________________________________

## Cache a function

```python
from pathlib import Path

from training.dal.cache import cached
from training.dal.dao.sqlite_dao import SqliteDao


@cached(version="1")
def slow_square(x: int) -> int:
    return x * x


with SqliteDao(path=Path("cache.sqlite")) as cache:
    slow_square(x=4, cache=cache)  # computes and stores 16
    slow_square(x=4, cache=cache)  # served from the store
slow_square(x=4)                   # no cache: calls straight through
```

The decorator adds a `cache` keyword to the function. The caller that opens the
database decides whether a call uses it, and the function body never sees a `Dao`
unless it declares a `cache` parameter itself, to pass on to other cached calls.

## What the key holds

The key is the sha256 of `module.qualname` and every argument. Primitives, bytes, and
lists, tuples, and dicts of them key by value, tagged by type so `"1"`, `1`, and
`(1,)` differ. An object keys by its `cache_key()`, which is how `self` of a class
keys a method. Any other argument raises `TypeError` at call time, because a key
built from something that cannot be compared would silently collide.

## One entry per key

A `Dao` stores the version next to the value, so a key never has two versions side by side. `get`
returns nothing when the key is missing or its stored version differs, so the caller
recomputes and `put` replaces the stale entry. `SqliteDao` logs a warning that names
the key and both versions when that happens.

The key never sees code. **Bump `version` whenever the function's output for the same
arguments changes**, including a fix in anything it calls. An unbumped fix keeps
serving the old value. A stored value that no longer unpickles, because a class was
renamed or moved, counts as a miss and is recomputed.

## Swap the store

`Dao` is a two-method protocol: `get(key, version)` and `put(key, value, version)`.
Anything with those methods, such as S3 or Lance, replaces `SqliteDao` without
touching a caller. `SqliteDao` uses the standard library only, switches the file to
WAL mode, and waits on a busy timeout, so parallel runs share one file without failing
on a lock. If two runs compute the same key, the later `put` wins and both values are
equal.
