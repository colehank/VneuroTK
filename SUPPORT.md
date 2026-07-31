# Support

VneuroTK is pre-alpha, volunteer-maintained software. There is no paid support program, service-level agreement, or guaranteed response time.

## Where to ask

- Use [GitHub issues](https://github.com/colehank/VneuroTK/issues) for reproducible bugs, focused feature proposals, and documentation problems.
- Use [GitHub private vulnerability reporting](https://github.com/colehank/VneuroTK/security/advisories/new) for suspected security vulnerabilities. Do not disclose exploitable details publicly.
- Do not use vulnerability reporting for general usage questions.

Before opening an issue, search existing issues and the [documentation](https://colehank.github.io/VneuroTK/). Include the VneuroTK and Python versions, operating system, installed extra and model backend, a minimal reproduction, and the complete traceback with secrets and private data removed. For data-related reports, identify the data source and applicable license or state that the information is unknown; never upload restricted participant data or stimuli. For network or integration behavior, explain how artifacts were obtained and whether the failure reproduces offline.

The issue templates request this information. Maintainers may close reports that cannot be reproduced, concern unsupported versions, lack information needed to investigate, or depend on third-party services or artifacts outside this project's control.

## Supported versions and scope

Development and fixes target the latest release and the current `main` branch. Older versions do not have a backport policy. Support is best-effort for the Python versions and optional extras declared in `pyproject.toml`; upstream models, datasets, hardware, drivers, hosted services, and their licenses remain the user's responsibility.

VneuroTK does not provide legal, research-ethics, clinical, or data-governance advice. See the [dataset provenance, licensing, citation, and ethics policy](docs/data-policy.md) before using or sharing research data.
