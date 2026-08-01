# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/colehank/vneurotk/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

vneurotk could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

To preview the docs locally:

```sh
just docs-serve
```

This starts Sphinx with live reload at http://localhost:8000. Changes under `docs/` and API docstrings under `src/` trigger a rebuild. MyST-NB renders checked-in notebook outputs without executing the notebooks.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/colehank/vneurotk/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `vneurotk` for local development.

1. Fork the `vneurotk` repo on GitHub.
2. Clone your fork locally:

   ```sh
   git clone git@github.com:your_name_here/vneurotk.git
   ```

3. Install your local copy with uv:

   ```sh
   cd vneurotk/
   uv sync
   ```

4. Create a branch for local development:

   ```sh
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and the tests:

   ```sh
   just qa
   ```

   Or run the tests alone:

   ```sh
   just test
   ```

6. Commit your changes and push your branch to GitHub:

   ```sh
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
3. The pull request should work for Python 3.11, 3.12, 3.13, and 3.14. Tests run in GitHub Actions on every pull request to the main branch; make sure the relevant lanes pass.
4. Identify affected optional extras and model backends. Document and explicitly gate network or full-integration tests so the default suite remains offline.
5. For datasets, stimuli, or model artifacts, provide authoritative provenance, license and citation requirements, and ethics/privacy implications. Do not include restricted data or invent licenses or DOIs; see the [data policy](data-policy.md).
6. State testable acceptance criteria and include the commands or other evidence used to verify them.

## Tips

To run a subset of tests:

```sh
uv run pytest tests/
```

## Releasing a New Version

Requires [GitHub CLI](https://cli.github.com/) (`gh`) installed and authenticated (`gh auth login`).

1. **Write the release notes:**
   Create a nonempty `CHANGELOG/<version>.md`, using the canonical version that
   will appear in the tag (for example, `CHANGELOG/1.2.3.md`).

2. **Commit and merge the release preparation to `main`:**
   ```bash
   git add CHANGELOG/
   git commit -m "Release <version>"
   ```
   The local `main` branch must have a clean working tree and exactly match
   `origin/main`; releasing from another, ahead, or behind branch is refused.

3. **Run the non-mutating rehearsal:**
   ```bash
   just release <version> --dry-run
   ```
   This fetches `origin/main`, checks local and remote tag absence, validates the
   lockfile, runs the default tests and `build-check`, then installs the wheel in
   a clean environment and checks its import and `importlib.metadata` version.
   It does not create a tag, a GitHub Release, or a PyPI upload.

4. **Optionally rehearse the upload client against TestPyPI:**
   ```bash
   SETUPTOOLS_SCM_PRETEND_VERSION=<version> just build-check
   just publish-test-dry-run
   ```
   `uv publish --dry-run` targets TestPyPI's upload and simple-index URLs without
   credentials and without uploading. This validates client-side file selection;
   it does **not** test TestPyPI authentication, Trusted Publishing, or server-side
   acceptance. Use a disposable version and a separately protected TestPyPI
   environment if a real TestPyPI upload is ever required.

5. **Create the release:**
   ```bash
   just release <version>   # e.g. just release 1.0.0
   ```
   After rerunning all preflight checks, the script makes one `gh release create`
   request targeting the validated commit. GitHub creates the tag and publishes
   the Release together; no standalone `git push` is used. A failed preflight
   therefore creates and pushes nothing. A transport/API failure can still have
   an indeterminate remote result, so inspect GitHub before retrying.

The published GitHub Release triggers `.github/workflows/publish.yml`. That
workflow checks out the exact release tag, rebuilds and validates one wheel and
one source distribution, verifies tag/version/filename/package metadata and a
clean installed-wheel import, and uploads those files as a run-scoped artifact.
The protected `pypi` environment then gates the publish job, which downloads that
exact artifact, attests it, and publishes via PyPI Trusted Publishing. A manual
workflow dispatch accepts only an existing canonical `v*` tag and passes through
the same protected environment; arbitrary tag pushes do not trigger publishing.

Repository administrators must configure required reviewers on the `pypi`
environment and configure PyPI Trusted Publishing for this workflow/environment.
GitHub workflow YAML cannot create or enforce those repository-side settings.
The package version is derived automatically from the git tag via `hatch-vcs`; do
not edit `pyproject.toml` for a release.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](code_of_conduct.md). By participating in this project you agree to abide by its terms.
