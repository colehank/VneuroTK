# /// script
# requires-python = ">=3.11"
# ///
"""Validate and create a GitHub Release for the current main commit."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_VERSION_RE = re.compile(
    r"^(?P<release>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:(?:[-.]?)(?P<pre>a|b|rc|alpha|beta|pre|preview)[-.]?(?P<pre_n>0|[1-9]\d*))?"
    r"(?:[.-]?post[-.]?(?P<post_n>0|[1-9]\d*))?"
    r"(?:[.-]?dev[-.]?(?P<dev_n>0|[1-9]\d*))?"
    r"(?:\+(?P<local>[a-z0-9]+(?:[.-][a-z0-9]+)*))?$",
    re.IGNORECASE,
)


class ReleaseError(RuntimeError):
    """A release precondition was not met."""


@dataclass(frozen=True)
class CommandResult:
    stdout: str = ""
    returncode: int = 0


class CommandRunner:
    """Small subprocess boundary that unit tests can replace deterministically."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> CommandResult:
        print(f"$ {shlex.join(command)}")  # noqa: T201
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=None if env is None else {**os.environ, **env},
            check=False,
            capture_output=capture_output,
            text=True,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if check and completed.returncode:
            detail = stderr.strip() or stdout.strip()
            suffix = f": {detail}" if detail else ""
            raise ReleaseError(f"command failed ({completed.returncode}): {shlex.join(command)}{suffix}")
        return CommandResult(stdout=stdout, returncode=completed.returncode)


def normalize_version(value: str) -> tuple[str, str]:
    """Return the version and its one-leading-v release tag."""
    version = value[1:] if value.startswith("v") else value
    match = _VERSION_RE.fullmatch(version)
    if not match:
        raise ReleaseError(
            "version must be a three-part PEP 440/semver-like version (for example 1.2.3, 1.2.3rc1, or 1.2.3-rc.1)"
        )
    parts = match.groupdict()
    canonical = f"{parts['release']}.{parts['minor']}.{parts['patch']}"
    if parts["pre"]:
        pre = parts["pre"].lower()
        pre = {"alpha": "a", "beta": "b", "pre": "rc", "preview": "rc"}.get(pre, pre)
        canonical += f"{pre}{parts['pre_n']}"
    if parts["post_n"]:
        canonical += f".post{parts['post_n']}"
    if parts["dev_n"]:
        canonical += f".dev{parts['dev_n']}"
    if parts["local"]:
        raise ReleaseError("local version identifiers (+...) cannot be published to public PyPI")
    return canonical, f"v{canonical}"


def validate_tag(value: str) -> tuple[str, str]:
    """Validate a canonical v-prefixed public release tag."""
    if not value.startswith("v"):
        raise ReleaseError("release tag must have exactly one leading v")
    version, tag = normalize_version(value)
    if value != tag:
        raise ReleaseError(f"release tag must be canonical: expected {tag!r}, found {value!r}")
    return version, tag


def is_prerelease(version: str) -> bool:
    """Return whether a public version is an alpha, beta, RC, or dev release."""
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ReleaseError(f"invalid canonical version: {version!r}")
    return match.group("pre") is not None or match.group("dev_n") is not None


def validate_publish(
    requested_tag: str,
    *,
    root: Path,
    runner: CommandRunner,
    event_name: str,
    event_path: Path | None = None,
) -> tuple[str, str]:
    """Validate a publish trigger and require its tag commit to be on origin/main."""
    version, tag = validate_tag(requested_tag)
    if event_name == "release":
        if event_path is None:
            raise ReleaseError("release event validation requires GITHUB_EVENT_PATH")
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        release_payload = payload.get("release", {})
        if release_payload.get("tag_name") != tag:
            raise ReleaseError("release payload/tag mismatch")
        if release_payload.get("draft") is not False or release_payload.get("published_at") is None:
            raise ReleaseError("release event must contain a published, non-draft Release")
    elif event_name != "workflow_dispatch":
        raise ReleaseError(f"unsupported publish event: {event_name!r}")

    runner.run(
        ("git", "fetch", "--force", "--tags", "origin", "+refs/heads/main:refs/remotes/origin/main"),
        cwd=root,
    )
    tag_commit = _output(runner, root, "git", "rev-list", "-n", "1", tag)
    if not tag_commit:
        raise ReleaseError(f"release tag does not resolve to a commit: {tag}")
    head_commit = _output(runner, root, "git", "rev-parse", "HEAD")
    if head_commit != tag_commit:
        raise ReleaseError(f"checkout/tag mismatch: HEAD {head_commit!r}, {tag} {tag_commit!r}")
    contained = runner.run(
        ("git", "merge-base", "--is-ancestor", tag_commit, "origin/main"),
        cwd=root,
        check=False,
    )
    if contained.returncode != 0:
        raise ReleaseError(f"release tag {tag} is not contained in origin/main")
    return version, tag


def emit_publish_outputs(version: str, tag: str, output_path: Path) -> None:
    """Append validated release values to the GitHub Actions output file."""
    with output_path.open("a", encoding="utf-8") as output:
        print(f"tag={tag}", file=output)
        print(f"version={version}", file=output)


def read_release_notes(root: Path, version: str) -> str:
    notes_path = root / "CHANGELOG" / f"{version}.md"
    if not notes_path.is_file():
        raise ReleaseError(f"release notes not found: {notes_path.relative_to(root)}")
    lines = notes_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    notes = "".join(lines).strip()
    if not notes:
        raise ReleaseError(f"release notes are empty: {notes_path.relative_to(root)}")
    return notes


def _output(runner: CommandRunner, root: Path, *command: str) -> str:
    return runner.run(command, cwd=root, capture_output=True).stdout.strip()


def run_preflight(
    version: str,
    tag: str,
    *,
    root: Path,
    runner: CommandRunner,
) -> tuple[str, str]:
    """Run every fallible validation before the release/tag API call."""
    notes = read_release_notes(root, version)

    if _output(runner, root, "git", "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReleaseError("working tree is not clean")

    branch = _output(runner, root, "git", "branch", "--show-current")
    if branch != "main":
        raise ReleaseError(f"release branch must be main, found {branch or 'detached HEAD'}")

    runner.run(("git", "fetch", "--prune", "origin", "main"), cwd=root)
    counts = _output(runner, root, "git", "rev-list", "--left-right", "--count", "origin/main...HEAD").split()
    if len(counts) != 2 or not all(part.isdigit() for part in counts):
        raise ReleaseError("could not determine ahead/behind state relative to origin/main")
    behind, ahead = (int(part) for part in counts)
    if behind or ahead:
        raise ReleaseError(f"main must exactly match origin/main (behind {behind}, ahead {ahead})")

    if _output(runner, root, "git", "tag", "--list", tag):
        raise ReleaseError(f"local tag already exists: {tag}")
    if _output(runner, root, "git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"):
        raise ReleaseError(f"remote tag already exists: {tag}")

    runner.run(("gh", "auth", "status"), cwd=root)
    runner.run(("just", "test"), cwd=root)

    build_env = {"SETUPTOOLS_SCM_PRETEND_VERSION": version}
    runner.run(("just", "build-check"), cwd=root, env=build_env)

    wheels = sorted((root / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ReleaseError(f"expected exactly one wheel in dist, found {len(wheels)}")

    with tempfile.TemporaryDirectory(prefix="vneurotk-release-") as temporary:
        smoke_root = Path(temporary)
        environment = smoke_root / "venv"
        python = environment / "bin" / "python"
        runner.run(("uv", "venv", "--python=3.13", str(environment)), cwd=smoke_root)
        runner.run(("uv", "pip", "install", "--python", str(python), str(wheels[0])), cwd=smoke_root)
        runner.run(
            (
                str(python),
                "-c",
                "import importlib.metadata as m, sys; "
                "expected=sys.argv[1]; actual=m.version('vneurotk'); "
                "assert actual == expected, f'{actual} != {expected}'; "
                "import vneurotk; print(f'installed-wheel smoke OK: vneurotk {actual}')",
                version,
            ),
            cwd=smoke_root,
        )

    commit = _output(runner, root, "git", "rev-parse", "HEAD")
    return notes, commit


def release(
    requested_version: str,
    *,
    dry_run: bool,
    root: Path,
    runner: CommandRunner,
) -> None:
    version, tag = normalize_version(requested_version)
    notes, commit = run_preflight(version, tag, root=root, runner=runner)
    if dry_run:
        print(f"Dry run passed: would create published GitHub Release {tag} at {commit}.")  # noqa: T201
        return

    # This single API operation creates the remote tag and published Release. Keeping
    # all validation before it avoids a pushed tag when a preflight check fails.
    command = [
        "gh",
        "release",
        "create",
        tag,
        "--target",
        commit,
        "--title",
        f"vneurotk {version}",
        "--notes",
        notes,
    ]
    if is_prerelease(version):
        command.append("--prerelease")
    runner.run(command, cwd=root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "validate-publish":
        parser = argparse.ArgumentParser(description="Validate a publish workflow trigger.")
        parser.add_argument("command")
        parser.add_argument("tag", help="canonical v-prefixed release tag")
        parser.add_argument("--event-name", required=True, choices=("release", "workflow_dispatch"))
        parser.add_argument("--event-path", type=Path)
        parser.add_argument("--github-output", type=Path)
        return parser.parse_args(arguments)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(command="release")
    parser.add_argument("version", help="release version, with an optional single leading v")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run all checks and builds without creating a tag or GitHub Release",
    )
    return parser.parse_args(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runner = CommandRunner()
        root = Path.cwd()
        if args.command == "validate-publish":
            version, tag = validate_publish(
                args.tag,
                root=root,
                runner=runner,
                event_name=args.event_name,
                event_path=args.event_path,
            )
            if args.github_output is not None:
                emit_publish_outputs(version, tag, args.github_output)
        else:
            release(args.version, dry_run=args.dry_run, root=root, runner=runner)
    except (ReleaseError, json.JSONDecodeError, OSError) as error:
        print(f"Error: {error}")  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
