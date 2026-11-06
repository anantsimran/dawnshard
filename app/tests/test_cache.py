import pytest
from training.dal.cache import cached
from training.dal.dao.sqlite_dao import SqliteDao


class _CountingCalls:
    def __init__(self):
        self.count = 0


calls = _CountingCalls()


@cached(version="1")
def _double(x):
    calls.count += 1
    return x * 2


@pytest.fixture(autouse=True)
def _reset_calls():
    calls.count = 0


@pytest.fixture
def dao(tmp_path):
    with SqliteDao(path=tmp_path / "c.sqlite") as opened:
        yield opened


def test_hit_skips_the_function(dao):
    assert _double(x=3, cache=dao) == 6
    assert _double(x=3, cache=dao) == 6
    assert calls.count == 1


def test_positional_and_keyword_arguments_share_a_key(dao):
    _double(3, cache=dao)  # noqa: NAR001
    _double(x=3, cache=dao)
    assert calls.count == 1


def test_different_argument_misses(dao):
    _double(x=3, cache=dao)
    _double(x=4, cache=dao)
    assert calls.count == 2


def test_argument_types_do_not_collide(dao):
    _double(x=1, cache=dao)
    _double(x="1", cache=dao)
    _double(x=(1,), cache=dao)
    _double(x=[1], cache=dao)
    assert calls.count == 4


def test_cache_none_calls_through_and_stores_nothing(dao):
    _double(x=3)
    _double(x=3)
    assert calls.count == 2


def test_version_bump_recomputes_and_replaces(dao):
    @cached(version="1")
    def versioned(x):
        calls.count += 1
        return x

    versioned(x=1, cache=dao)

    # A second function object with the same module and qualname stands for the
    # same function after its version was bumped.
    @cached(version="2")
    def versioned(x):  # noqa: F811
        calls.count += 1
        return x

    versioned(x=1, cache=dao)
    versioned(x=1, cache=dao)
    assert calls.count == 2


def test_unkeyable_argument_raises(dao):
    with pytest.raises(TypeError):
        _double(x=object(), cache=dao)


def test_unpicklable_stored_value_recomputes(dao):
    @cached(version="1")
    def make(x):
        calls.count += 1
        return x

    make(x=1, cache=dao)
    # A protocol-0 pickle that names a module that does not exist, as a moved class would.
    key = dao._connection.execute("SELECT key FROM cache").fetchone()[0]  # noqa: NAR001
    broken = b"cnot_a_module\nFoo\n."
    dao.put(key=key, value=broken, version="1")

    assert make(x=1, cache=dao) == 1
    assert calls.count == 2
    assert make(x=1, cache=dao) == 1
    assert calls.count == 2


class _Keyed:
    def __init__(self, key):
        self.key = key

    def cache_key(self):
        return self.key

    @cached(version="1")
    def method(self, x):
        calls.count += 1
        return x


def test_method_keys_by_cache_key_of_self(dao):
    _Keyed(key="a").method(x=1, cache=dao)
    _Keyed(key="a").method(x=1, cache=dao)
    assert calls.count == 1
    _Keyed(key="b").method(x=1, cache=dao)
    assert calls.count == 2


def test_function_that_declares_cache_receives_it(dao):
    seen = []

    @cached(version="1")
    def forwards(x, cache=None):
        seen.append(cache)
        return x

    forwards(x=1, cache=dao)
    forwards(x=1)
    assert seen == [dao, None]
