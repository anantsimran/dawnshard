"""expect.py -- unwrap an Optional in one line, raising if it holds None."""


def expect[T](value: T | None, message: str) -> T:
    """Return `value`, narrowed from `T | None` to `T`.

    Unlike `assert value is not None`, the check survives `python -O`.

    Raises:
        ValueError: With `message`, if `value` is None.
    """
    if value is None:
        raise ValueError(message)  # noqa: NAR001
    return value
