import sqlite3

import pytest
from training.dal.dao.sqlite_dao import SqliteDao


def test_missing_key_gives_none(tmp_path):
    with SqliteDao(path=tmp_path / "c.sqlite") as dao:
        assert dao.get(key="k", version="1") is None


def test_put_then_get_roundtrips(tmp_path):
    with SqliteDao(path=tmp_path / "c.sqlite") as dao:
        dao.put(key="k", value=b"value", version="1")
        assert dao.get(key="k", version="1") == b"value"


def test_version_mismatch_gives_none(tmp_path):
    with SqliteDao(path=tmp_path / "c.sqlite") as dao:
        dao.put(key="k", value=b"value", version="1")
        assert dao.get(key="k", version="2") is None


def test_put_replaces_the_entry_and_keeps_one_row(tmp_path):
    path = tmp_path / "c.sqlite"
    with SqliteDao(path=path) as dao:
        dao.put(key="k", value=b"old", version="1")
        dao.put(key="k", value=b"new", version="2")
        assert dao.get(key="k", version="1") is None
        assert dao.get(key="k", version="2") == b"new"
    rows = sqlite3.connect(path).execute("SELECT COUNT(*) FROM cache").fetchone()  # noqa: NAR001
    assert rows == (1,)


def test_value_survives_a_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    with SqliteDao(path=path) as dao:
        dao.put(key="k", value=b"value", version="1")
    with SqliteDao(path=path) as dao:
        assert dao.get(key="k", version="1") == b"value"


def test_creates_the_parent_directory(tmp_path):
    with SqliteDao(path=tmp_path / "nested" / "c.sqlite") as dao:
        dao.put(key="k", value=b"v", version="1")
    assert (tmp_path / "nested" / "c.sqlite").exists()


def test_use_after_close_raises(tmp_path):
    dao = SqliteDao(path=tmp_path / "c.sqlite")
    dao.close()
    with pytest.raises(sqlite3.ProgrammingError):
        dao.get(key="k", version="1")
