"""`@cached`: store a function's result in a `Dao`, keyed by its arguments.

```python
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

The key is the sha256 of the function's `module.qualname` and every argument. Primitives
and bytes, and lists, tuples, and dicts of them, key by value. An object keys by its
`cache_key()` (see `CacheKeyed`), which is how `self` of a class keys a method. Any
other argument raises `TypeError`, rather than keying on something that would collide.

The key never sees code. Bump `version` whenever the function's output for the same
arguments changes, including a fix in anything it calls; otherwise the stale value is
served. A stored value that no longer unpickles, because a class was renamed or moved,
counts as a miss.
"""

import functools
import hashlib
import inspect
import pickle
from typing import Any, Callable, Protocol, runtime_checkable

from loguru import logger
from training.dal.dao.dao import Dao

# A class rename or move surfaces as one of these while unpickling.
_UNPICKLE_ERRORS = (pickle.UnpicklingError, AttributeError, ModuleNotFoundError)


@runtime_checkable
class CacheKeyed(Protocol):
    """An object that can stand in a cache key, such as a tokenizer."""

    def cache_key(self) -> str:
        """Return a string that is equal for two objects exactly when they behave the same."""
        ...


def _key_part(value: Any) -> str:
    """Map an argument to a canonical string, tagged by type so "1" and 1 differ.

    Raises:
        TypeError: If `value` is not a primitive, bytes, a list, tuple, or dict of
            keyable values, or a `CacheKeyed` object.
    """
    if value is None or isinstance(value, (bool, int, float, str)):  # noqa: NAR001
        return f"{type(value).__name__}:{value!r}"  # noqa: NAR001
    if isinstance(value, bytes):  # noqa: NAR001
        return f"bytes:{value.hex()}"
    if isinstance(value, (list, tuple)):  # noqa: NAR001
        parts = ",".join(_key_part(value=item) for item in value)  # noqa: NAR001
        return f"{type(value).__name__}[{parts}]"  # noqa: NAR001
    if isinstance(value, dict):  # noqa: NAR001
        items = sorted(  # noqa: NAR001
            (_key_part(value=k), _key_part(value=v)) for k, v in value.items()
        )
        return "dict{" + ",".join(f"{k}=>{v}" for k, v in items) + "}"  # noqa: NAR001
    if isinstance(value, CacheKeyed):  # noqa: NAR001
        return f"{type(value).__name__}:{value.cache_key()}"  # noqa: NAR001
    raise TypeError(  # noqa: NAR001
        f"cannot build a cache key from a {type(value).__name__}; pass primitives, "  # noqa: NAR001
        "bytes, containers of them, or an object with cache_key()"
    )


def cached[R](version: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Return a decorator that caches a function's result in a `Dao`.

    The wrapped callable takes an extra keyword, `cache: Dao | None = None`. With
    None it calls through and stores nothing. If the function itself declares a
    `cache` parameter, it receives the same value, so it can pass it on to other
    cached calls. `cache` is never part of the key.

    Args:
        version: Stored next to the value, not in the key. A stored entry with a
            different version is a miss, and the recomputed value replaces it.

    Side effects:
        On a miss, pickles the result and `put`s it. Logs a hit or a miss at INFO.
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        """Wrap `func` with the cache lookup."""
        signature = inspect.signature(obj=func)
        takes_cache = "cache" in signature.parameters  # noqa: NAR001
        name = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(wrapped=func)
        def wrapper(*args: Any, cache: Dao | None = None, **kwargs: Any) -> R:
            """Call `func`, going through `cache` when one is given."""
            if takes_cache:
                kwargs["cache"] = cache
            if cache is None:
                return func(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = {k: v for k, v in bound.arguments.items() if k != "cache"}  # noqa: NAR001
            key = hashlib.sha256(
                string=(name + "|" + _key_part(value=arguments)).encode(encoding="utf-8")
            ).hexdigest()
            stored = cache.get(key=key, version=version)
            if stored is not None:
                try:
                    result = pickle.loads(stored)  # noqa: NAR001
                except _UNPICKLE_ERRORS as error:
                    logger.warning(  # noqa: NAR001
                        "cache entry for {} does not unpickle ({}); recomputing", name, error
                    )
                else:
                    logger.info("cache hit: {} ({})", name, key[:12])  # noqa: NAR001
                    return result
            else:
                logger.info("cache miss: {} ({})", name, key[:12])  # noqa: NAR001
            result = func(*args, **kwargs)
            cache.put(key=key, value=pickle.dumps(obj=result), version=version)
            return result

        return wrapper

    return decorator
