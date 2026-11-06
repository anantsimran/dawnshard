"""The storage contract `@cached` talks to.

A `Dao` holds one entry per key, and the entry carries the version of the code that
wrote it. `get` answers only for the version the caller asks for, so a caller that
bumped its version recomputes and `put` replaces the stale entry. No store keeps two
versions of one key side by side.

`SqliteDao` is the implementation today. Any store with these two methods, such as S3
or Lance, drops in without touching a caller.
"""

from typing import Protocol


class Dao(Protocol):
    """Versioned key-value storage: one entry per key, replaced on `put`."""

    def get(self, key: str, version: str) -> bytes | None:
        """Return the stored value, or None if the key is missing or has another version."""
        ...

    def put(self, key: str, value: bytes, version: str) -> None:
        """Store `value` under `key` with `version`, replacing any existing entry."""
        ...
