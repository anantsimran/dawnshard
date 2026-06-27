.PHONY: test precheck docker-build docker-build-dev docker-run docker-run-dev

IMAGE_NAME := dawnshard

# Run tests. Optionally scope to a directory: make test DIR=app/tests/foo
test:
	uv run pytest $(if $(DIR),$(DIR),app/tests/)

# Run type checking and linting (pyright + ruff check + ruff format)
precheck:
	uv run pyright
	uv run ruff check app/src/
	uv run ruff format app/src/
	uv run mdformat .

# Build the production Docker image
docker-build:
	docker build -t $(IMAGE_NAME):latest .

# Build the dev Docker image (includes dev dependencies)
docker-build-dev:
	docker build --build-arg DEV=true -t $(IMAGE_NAME):dev .

# Run a shell in the production image
docker-run:
	docker run -it $(IMAGE_NAME):latest bash

# Run a shell in the dev image with the app directory mounted for live editing
docker-run-dev:
	docker run -it \
		-v $(PWD)/app:/app \
		$(IMAGE_NAME):dev bash
