.PHONY: test precheck check-named-args docker-build docker-build-dev docker-run docker-run-dev docker-build-dev-test nb-to-py py-to-nb

IMAGE_NAME := dawnshard

# Run tests. Optionally scope to a directory: make test DIR=app/tests/foo
test:
	uv run pytest $(if $(DIR),$(DIR),app/tests/)

# Run type checking and linting (pyright + ruff check + ruff format)
precheck:
	@uv run pyright
	@uv run ruff check app/src/ --fix
	@uv run ruff format app/src/
	@uv run mdformat . ; \
	if ! git diff --quiet; then \
		echo "\033[0;31mmdformat reformatted files. Stage the changes and run precheck again.\033[0m"; \
		exit 1; \
	fi

# Check all source files for positional arguments (run before opening a PR)
check-named-args:
	@find app/src -name "*.py" | xargs uv run python scripts/check_named_args.py

# Build the production Docker image
docker-build:
	docker build -t $(IMAGE_NAME):latest .

# Build the dev Docker image (includes dev dependencies)
docker-build-dev:
	docker build --build-arg DEV=true -t $(IMAGE_NAME):dev .

# Run a shell in the production image
docker-run:
	docker run -it $(IMAGE_NAME):latest bash

# Build the dev Docker image and run the test suite inside it
docker-build-dev-test:
	docker build --build-arg DEV=true -t $(IMAGE_NAME):dev .
	docker run --rm $(IMAGE_NAME):dev uv run pytest

# Run a shell in the dev image with the app directory mounted for live editing
docker-run-dev:
	docker run -it \
		-v $(PWD)/app:/app \
		$(IMAGE_NAME):dev bash

## Jupyter ↔ Python Conversion
# Usage: make nb-to-py NB=path/to/notebook.ipynb
nb-to-py:
	@test -n "$(NB)" || (echo "Usage: make nb-to-py NB=path/to/notebook.ipynb" && exit 1)
	uv run jupyter nbconvert --to script $(NB)

# Usage: make py-to-nb PY=path/to/script.py
py-to-nb:
	@test -n "$(PY)" || (echo "Usage: make py-to-nb PY=path/to/script.py" && exit 1)
	uv run jupytext --to notebook $(PY)
