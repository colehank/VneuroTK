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

# Check formatting, lint, types, and the default offline test lane
qa:
    uv run --python=3.13 --group dev ruff format --check .
    uv run --python=3.13 --group dev ruff check .
    just type-check-concise
    uv run --python=3.13 --group dev pytest

# Apply formatting and safe lint fixes
format:
    uv run --python=3.13 --group lint ruff format .

fix:
    uv run --python=3.13 --group lint ruff check . --fix

# Offline/lightweight test lanes
test-core *ARGS:
    uv run --python=3.13 --group test pytest {{ARGS}}

# Published sample archive lanes (offline contract plus explicit real-data gates)
test-sample-integrity *ARGS:
    uv run --python=3.13 --group test pytest tests/test_datasets_sample.py {{ARGS}}

test-sample-nod *ARGS:
    VNEUROTK_TEST_SAMPLE_DATA=1 VNEUROTK_TEST_INTEGRATION=1 VNEUROTK_RUN_NETWORK=1 VNEUROTK_TEST_SLOW=1 uv run --python=3.13 --group test --extra mne pytest -o addopts="--strict-config --strict-markers -ra" -m "sample_data and integration and network and slow" tests/test_sample_data_integration.py -k "nod" {{ARGS}}

test-sample-monkey *ARGS:
    VNEUROTK_TEST_SAMPLE_DATA=1 VNEUROTK_TEST_INTEGRATION=1 VNEUROTK_RUN_NETWORK=1 VNEUROTK_TEST_SLOW=1 uv run --python=3.13 --group test pytest -o addopts="--strict-config --strict-markers -ra" -m "sample_data and integration and network and slow" tests/test_sample_data_integration.py -k "monkey" {{ARGS}}

test-samples *ARGS:
    just test-sample-integrity {{ARGS}}
    just test-sample-nod {{ARGS}}
    just test-sample-monkey {{ARGS}}

test-vision *ARGS:
    VNEUROTK_TEST_VISION=1 uv run --python=3.13 --group test --extra vision pytest -m "vision and not backend_transformers and not backend_timm and not backend_thingsvision and not integration and not network and not slow" {{ARGS}}

test-viz *ARGS:
    VNEUROTK_TEST_VIZ=1 uv run --python=3.13 --group test --extra viz pytest -m "viz and not integration and not network and not slow" {{ARGS}}

# Backend-specific lanes install and validate their matching extra
test-backend-transformers *ARGS:
    VNEUROTK_TEST_VISION=1 VNEUROTK_TEST_BACKEND_TRANSFORMERS=1 uv run --python=3.13 --group test --extra vision pytest -m "backend_transformers and not integration and not network and not slow" {{ARGS}}

test-backend-timm *ARGS:
    VNEUROTK_TEST_VISION=1 VNEUROTK_TEST_BACKEND_TIMM=1 uv run --python=3.13 --group test --extra timm pytest -m "backend_timm and not integration and not network and not slow" {{ARGS}}

test-backend-thingsvision *ARGS:
    VNEUROTK_TEST_VISION=1 VNEUROTK_TEST_BACKEND_THINGSVISION=1 uv run --python=3.12 --group test --extra thingsvision pytest -m "backend_thingsvision and not integration and not network and not slow" {{ARGS}}

test-backends *ARGS:
    just test-backend-transformers {{ARGS}}
    just test-backend-timm {{ARGS}}
    just test-backend-thingsvision {{ARGS}}

# Network tests remain explicit even inside integration runs
test-network *ARGS:
    VNEUROTK_TEST_VISION=1 VNEUROTK_TEST_BACKEND_TRANSFORMERS=1 VNEUROTK_TEST_INTEGRATION=1 VNEUROTK_RUN_NETWORK=1 VNEUROTK_TEST_SLOW=1 uv run --python=3.13 --group test --extra vision pytest -m "network" {{ARGS}}

test-integration *ARGS:
    VNEUROTK_TEST_VISION=1 VNEUROTK_TEST_VIZ=1 VNEUROTK_TEST_BACKEND_TRANSFORMERS=1 VNEUROTK_TEST_BACKEND_TIMM=1 VNEUROTK_TEST_HDF5_COMPAT=1 VNEUROTK_TEST_INTEGRATION=1 VNEUROTK_TEST_SLOW=1 uv run --python=3.13 --group test --extra viz --extra timm pytest -m "(hdf5_compat or integration or slow) and not backend_thingsvision and not network" {{ARGS}}
    VNEUROTK_TEST_VISION=1 VNEUROTK_TEST_BACKEND_THINGSVISION=1 VNEUROTK_TEST_INTEGRATION=1 VNEUROTK_TEST_SLOW=1 uv run --python=3.12 --group test --extra thingsvision pytest -m "backend_thingsvision and (integration or slow) and not network" {{ARGS}}

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

# Validate repository policy metadata
metadata-check:
    uvx --from cffconvert==2.0.0 cffconvert --validate
    @uv run --frozen --python=3.13 --group packaging python -c 'import pathlib,tomllib; import yaml; root=pathlib.Path("."); tomllib.loads((root / "zensical.toml").read_text()); files=list((root / ".github").rglob("*.yml")) + list((root / ".github").rglob("*.yaml")); [yaml.safe_load(path.read_text()) for path in files]; print(f"metadata OK: zensical.toml and {len(files)} YAML files")'

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

# Build and validate wheel and source distribution artifacts
build-check:
    uv lock --check
    just metadata-check
    just build
    uv run --frozen --python=3.13 --group packaging twine check dist/*
    uv run --frozen --python=3.13 --group packaging check-wheel-contents dist/*.whl
    @uv run --frozen --python=3.13 python -c 'import pathlib,tarfile,zipfile; dist=pathlib.Path("dist"); wheel=next(dist.glob("*.whl")); sdist=next(dist.glob("*.tar.gz")); wz=zipfile.ZipFile(wheel); st=tarfile.open(sdist); wn=wz.namelist(); sn=st.getnames(); root=pathlib.PurePosixPath(sn[0]).parts[0]; required=lambda names,prefix: all((any(n.startswith(prefix) for n in names), f"{prefix}py.typed" in names, any(n.endswith("/LICENSE") or n == "LICENSE" for n in names), any(n.endswith("/README.md") or n == "README.md" for n in names), any(n.endswith("/CITATION.cff") or n == "CITATION.cff" for n in names))); forbidden_parts={".github","build","dist","docs","notebooks","site","tests","fixtures","__pycache__"}; forbidden_suffixes={".db",".ipynb",".pyc",".pyo",".sqlite",".sqlite3"}; bad=lambda names: [n for n in names if forbidden_parts.intersection(pathlib.PurePosixPath(n).parts) or pathlib.PurePosixPath(n).suffix.lower() in forbidden_suffixes or ".sqlite" in pathlib.PurePosixPath(n).name.lower()]; assert required(wn,"vneurotk/"), f"{wheel.name}: required contents missing"; assert required(sn,f"{root}/src/vneurotk/"), f"{sdist.name}: required contents missing"; assert not bad(wn), f"{wheel.name}: forbidden contents: {bad(wn)}"; assert not bad(sn), f"{sdist.name}: forbidden contents: {bad(sn)}"; print(f"artifact contents OK: {wheel.name}, {sdist.name}")'

# Validate locally, then atomically create a published GitHub Release and tag
# (usage: just release 1.0.0; add --dry-run to run every preflight only)
release version *ARGS:
    uv run scripts/release.py {{version}} {{ARGS}}

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

# Rehearse the final upload client against TestPyPI without uploading
# (requires artifacts from `just build-check`; this command has no credentials)
publish-test-dry-run:
    uv publish --dry-run --publish-url https://test.pypi.org/legacy/ --check-url https://test.pypi.org/simple/ dist/*
