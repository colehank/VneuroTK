# Security Policy

## Reporting a Vulnerability

If you find a security vulnerability in vneurotk, please report it through [GitHub's private vulnerability reporting](https://github.com/colehank/vneurotk/security/advisories/new). This keeps the details private while we work on a fix.

Please include:

- What you found and how to reproduce it
- Which version you're using
- Any relevant logs or output (redact secrets)

## Security Measures

This project ships with security hardening out of the box:

- **CodeQL** scans code for injection, SSRF, path traversal, and other dataflow vulnerabilities using the `security-extended` query suite
- **Zizmor** audits GitHub Actions workflows for excessive permissions, unpinned actions, credential exposure, and cache poisoning risks
- **Dependabot** updates locked Python dependencies monthly and keeps GitHub Actions pinned by SHA current through weekly PRs, with a 7-day cooldown before newly available versions are proposed
- **All actions pinned by SHA** with version comments, not floating tags
- **Minimal workflow permissions** (`permissions: {}` at the top level, scoped per job)
- **`persist-credentials: false`** on checkout steps to prevent token leakage

## Response and Disclosure

This is a volunteer-maintained open-source project. Reports are reviewed on a best-effort basis; acknowledgment, remediation, and disclosure timelines are not guaranteed. Please keep vulnerability details private until the maintainer has assessed the report and coordinated disclosure with you.

If private vulnerability reporting is unavailable, open a public issue that asks the maintainer to enable a private contact channel, without including vulnerability details.

## Supported Versions

Security fixes are applied to the latest release on the `main` branch. There is no backport policy for older versions.
