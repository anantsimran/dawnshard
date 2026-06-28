# CLAUDE.md

## Named-argument linting

All function calls must use keyword arguments. A custom checker (`scripts/check_named_args.py`) enforces this as part of `make precheck`.

**Suppress with `# noqa: NAR001`** when keyword arguments are genuinely unavailable:

- C builtins and C extension methods: `print`, `len`, `set`, `range`, `iter`, `next`, `isinstance`, `sorted`, `zip`, `str`, `type`, `list.append`, `dict.update`, etc.
- `nn.Module.__call__` (PyTorch forward pass): `model(x)`, `criterion(pred, target)`
- `nn.Sequential(*modules)` and other `*args`-based constructors
- `Callable` type aliases where parameter names are not part of the contract
- loguru `logger.info(msg, *args)` and similar format-style logging

**Do not suppress** calls to regular Python functions and methods that have named parameters — use keyword arguments instead.

The ruff linter is configured to allow `NAR001` as an external code via `pyproject.toml`:

```toml
[tool.ruff.lint]
external = ["NAR001"]
```

## Line length

The project enforces an 88-character line limit (configured in `pyproject.toml`). `ruff format` (run by `make precheck`) handles most wrapping automatically, but **cannot** split lines that contain `# noqa` comments or long string literals — those must be wrapped manually.

The standard pattern is to open a parenthesis and move the `# noqa` to that first line:

```python
# Before (too long):
some_call(arg1, arg2, arg3)  # noqa: NAR001

# After:
some_call(  # noqa: NAR001
    arg1,
    arg2,
    arg3,
)
```
