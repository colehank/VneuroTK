# Justfile for vneurotk

# Show available commands
list:
    @just --list

alias b := build
alias c := clean
alias d := docs-serve
alias t := test
alias tc := type-check

# Type check the project with ty
type-check:
    uv run --python=3.13 --group typecheck ty check .

# Type check with concise output (one diagnostic per line)
type-check-concise:
    uv run --python=3.13 --group typecheck ty check --output-format=concise .

# Type check in watch mode (rechecks on file changes)
type-check-watch:
    uv run --python=3.13 --group typecheck ty check --watch .

# Run all the formatting, linting, and testing commands
qa:
    uv run --python=3.13 --group dev ruff format .
    uv run --python=3.13 --group dev ruff check . --fix
    uv run --python=3.13 --group dev ruff check --select I --fix .
    uv run --python=3.13 --group dev ty check --output-format=concise .
    uv run --python=3.13 --group dev pytest .

# Run all the tests for all the supported Python versions
testall:
    uv run --python=3.12 --group test pytest
    uv run --python=3.13 --group test pytest
    uv run --python=3.14 --group test pytest

# Run all the tests, but allow for arguments to be passed
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.13 --group test pytest {{ARGS}}

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.13 --group test pytest --pdb --maxfail=10 {{ARGS}}

# Run tests with coverage across all supported Python versions
coverage:
    uv run --python=3.12 --group test python -m coverage run -m pytest
    uv run --python=3.13 --group test python -m coverage run -m pytest
    uv run --python=3.14 --group test python -m coverage run -m pytest
    uv run --python=3.13 --group test python -m coverage combine
    uv run --python=3.13 --group test python -m coverage report
    uv run --python=3.13 --group test python -m coverage html

# Serve docs locally with live reload
docs-serve:
    -lsof -ti :8000 | xargs kill
    uv run --group docs zensical serve --dev-addr 0.0.0.0:8000

# Build docs (strict mode, fails on warnings)
docs-build:
    uv run --group docs zensical build --clean

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

# Tag, push, and create a GitHub release (usage: just release 1.0.0)
release version:
    uv run scripts/release.py {{version}}

# Remove all build, test, coverage and Python artifacts
clean:
	clean-build
	clean-pyc
	clean-test

# Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# Remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

# Remove test and coverage artifacts
clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/
	rm -fr .pytest_cache

# Publish to PyPI (manual alternative to GitHub Actions)
publish:
    uv build
    uv publish
