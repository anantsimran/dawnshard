"""A `Dao` on one SQLite file, using only the standard library."""

import sqlite3
from pathlib import Path
from types import TracebackType

from loguru import logger

BUSY_TIMEOUT_SECONDS = 60.0


class SqliteDao:
    """A `Dao` backed by one table, `cache(key TEXT PRIMARY KEY, version TEXT, value BLOB)`.

    Open it with `with SqliteDao(path=...) as cache:` so the connection closes when
    the run ends. WAL mode and a busy timeout let parallel runs share one file
    without failing on a lock; if two runs compute the same key, the later `put`
    wins and both values are equal.
    """

    def __init__(self, path: Path) -> None:
        """Open (and create, with its parent directory) the database at `path`.

        Side effects:
            Creates `path`'s parent directory and the `cache` table if missing, and
            switches the database to WAL mode.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(  # noqa: NAR001
            path, timeout=BUSY_TIMEOUT_SECONDS
        )
        self._connection.execute("PRAGMA journal_mode=WAL")  # noqa: NAR001
        self._connection.execute(  # noqa: NAR001
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, version TEXT NOT NULL, value BLOB NOT NULL)"
        )
        self._connection.commit()

    def get(self, key: str, version: str) -> bytes | None:
        """Return the value for `key` if it was stored under `version`, else None.

        Side effects:
            Logs a warning naming the key and both versions when the key exists
            under a different version.
        """
        row = self._connection.execute(  # noqa: NAR001
            "SELECT version, value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        stored_version, value = row
        if stored_version != version:
            logger.warning(  # noqa: NAR001
                "cache entry {} was stored under version {}, not {}; it will be recomputed",
                key,
                stored_version,
                version,
            )
            return None
        return value

    def put(self, key: str, value: bytes, version: str) -> None:
        """Store `value` under `key` and `version`, replacing the old entry."""
        with self._connection:
            self._connection.execute(  # noqa: NAR001
                "INSERT OR REPLACE INTO cache (key, version, value) VALUES (?, ?, ?)",
                (key, version, value),
            )

    def close(self) -> None:
        """Close the connection; later calls raise `sqlite3.ProgrammingError`."""
        self._connection.close()

    def __enter__(self) -> "SqliteDao":
        """Return self, so `with SqliteDao(path=...) as cache:` works."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection; exceptions propagate."""
        self.close()
