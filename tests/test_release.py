from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("release_script", ROOT / "scripts" / "release.py")
assert SPEC and SPEC.loader
release_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_script
SPEC.loader.exec_module(release_script)

CommandResult = release_script.CommandResult
ReleaseError = release_script.ReleaseError


class FakeRunner:
    def __init__(self, outputs: dict[tuple[str, ...], str] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []

    def run(self, command, *, cwd, env=None, check=True, capture_output=False):
        command = tuple(command)
        self.calls.append((command, env))
        return CommandResult(stdout=self.outputs.get(command, ""))


def _successful_runner() -> FakeRunner:
    return FakeRunner(
        {
            ("git", "branch", "--show-current"): "main\n",
            ("git", "rev-list", "--left-right", "--count", "origin/main...HEAD"): "0\t0\n",
            ("git", "rev-parse", "HEAD"): "abc123\n",
        }
    )


def _prepare_release(tmp_path: Path, version: str = "1.2.3") -> None:
    notes = tmp_path / "CHANGELOG" / f"{version}.md"
    notes.parent.mkdir()
    notes.write_text(f"# {version}\n\nRelease notes.\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / f"vneurotk-{version}-py3-none-any.whl").touch()


@pytest.mark.parametrize(
    ("requested", "version", "tag"),
    [
        ("1.2.3", "1.2.3", "v1.2.3"),
        ("v1.2.3", "1.2.3", "v1.2.3"),
        ("1.2.3rc1", "1.2.3rc1", "v1.2.3rc1"),
        ("1.2.3-rc.1", "1.2.3rc1", "v1.2.3rc1"),
    ],
)
def test_normalize_version(requested, version, tag):
    assert release_script.normalize_version(requested) == (version, tag)


@pytest.mark.parametrize(
    "requested",
    ["", "v", "vv1.2.3", "1.2", "01.2.3", "1.2.3.4", "1.2.3 nope", "1.2.3+linux.1"],
)
def test_normalize_version_rejects_invalid_or_local_values(requested):
    with pytest.raises(ReleaseError):
        release_script.normalize_version(requested)


def test_empty_release_notes_fail_before_commands(tmp_path):
    notes = tmp_path / "CHANGELOG" / "1.2.3.md"
    notes.parent.mkdir()
    notes.write_text("# 1.2.3\n\n", encoding="utf-8")
    runner = FakeRunner()

    with pytest.raises(ReleaseError, match="empty"):
        release_script.release("1.2.3", dry_run=True, root=tmp_path, runner=runner)

    assert runner.calls == []


def test_dirty_tree_stops_before_fetch_or_mutation(tmp_path):
    _prepare_release(tmp_path)
    runner = FakeRunner({("git", "status", "--porcelain=v1", "--untracked-files=all"): " M file\n"})

    with pytest.raises(ReleaseError, match="not clean"):
        release_script.release("1.2.3", dry_run=False, root=tmp_path, runner=runner)

    assert [call for call, _ in runner.calls] == [("git", "status", "--porcelain=v1", "--untracked-files=all")]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({("git", "branch", "--show-current"): "feature\n"}, "must be main"),
        (
            {
                ("git", "branch", "--show-current"): "main\n",
                ("git", "rev-list", "--left-right", "--count", "origin/main...HEAD"): "1 0\n",
            },
            "behind 1, ahead 0",
        ),
        (
            {
                ("git", "branch", "--show-current"): "main\n",
                ("git", "rev-list", "--left-right", "--count", "origin/main...HEAD"): "0 1\n",
            },
            "behind 0, ahead 1",
        ),
        (
            {
                ("git", "branch", "--show-current"): "main\n",
                ("git", "rev-list", "--left-right", "--count", "origin/main...HEAD"): "0 0\n",
                ("git", "tag", "--list", "v1.2.3"): "v1.2.3\n",
            },
            "local tag already exists",
        ),
        (
            {
                ("git", "branch", "--show-current"): "main\n",
                ("git", "rev-list", "--left-right", "--count", "origin/main...HEAD"): "0 0\n",
                ("git", "ls-remote", "--tags", "origin", "refs/tags/v1.2.3"): "hash\trefs/tags/v1.2.3\n",
            },
            "remote tag already exists",
        ),
    ],
)
def test_preflight_failures_never_create_release(tmp_path, overrides, message):
    _prepare_release(tmp_path)
    runner = FakeRunner(overrides)

    with pytest.raises(ReleaseError, match=message):
        release_script.release("1.2.3", dry_run=False, root=tmp_path, runner=runner)

    assert not any(command[:3] == ("gh", "release", "create") for command, _ in runner.calls)


def test_dry_run_executes_all_checks_but_creates_nothing(tmp_path):
    _prepare_release(tmp_path)
    runner = _successful_runner()

    release_script.release("v1.2.3", dry_run=True, root=tmp_path, runner=runner)

    commands = [command for command, _ in runner.calls]
    assert ("git", "fetch", "--prune", "origin", "main") in commands
    assert ("just", "test") in commands
    assert ("just", "build-check") in commands
    assert any(command[:3] == ("uv", "pip", "install") for command in commands)
    smoke = next(command for command in commands if command and command[0].endswith("/bin/python"))
    assert smoke[-1] == "1.2.3"
    assert "importlib.metadata" in smoke[2]
    assert "import vneurotk" in smoke[2]
    assert not any(command[:3] == ("gh", "release", "create") for command in commands)


def test_release_mutation_is_last_and_targets_validated_commit(tmp_path):
    _prepare_release(tmp_path)
    runner = _successful_runner()

    release_script.release("1.2.3", dry_run=False, root=tmp_path, runner=runner)

    command = runner.calls[-1][0]
    assert command[:3] == ("gh", "release", "create")
    assert command[3] == "v1.2.3"
    assert command[command.index("--target") + 1] == "abc123"
    assert command[command.index("--notes") + 1] == "Release notes."


def test_prerelease_release_is_marked_on_github(tmp_path):
    _prepare_release(tmp_path, "1.2.3rc1")
    runner = _successful_runner()

    release_script.release("1.2.3rc1", dry_run=False, root=tmp_path, runner=runner)

    command = runner.calls[-1][0]
    assert command[:4] == ("gh", "release", "create", "v1.2.3rc1")
    assert "--prerelease" in command


@pytest.mark.parametrize("version", ["1.2.3a1", "1.2.3b1", "1.2.3rc1", "1.2.3.dev1"])
def test_prerelease_policy(version):
    assert release_script.is_prerelease(version)


@pytest.mark.parametrize("version", ["1.2.3", "1.2.3.post1"])
def test_final_and_post_releases_are_not_prereleases(version):
    assert not release_script.is_prerelease(version)


def _publish_runner(*, tag_commit="abc123", head_commit="abc123", ancestry_returncode=0):
    class PublishRunner(FakeRunner):
        def run(self, command, *, cwd, env=None, check=True, capture_output=False):
            result = super().run(command, cwd=cwd, env=env, check=check, capture_output=capture_output)
            if tuple(command[:3]) == ("git", "merge-base", "--is-ancestor"):
                return CommandResult(returncode=ancestry_returncode)
            return result

    return PublishRunner(
        {
            ("git", "rev-list", "-n", "1", "v1.2.3"): f"{tag_commit}\n",
            ("git", "rev-parse", "HEAD"): f"{head_commit}\n",
        }
    )


def test_workflow_dispatch_rejects_off_main_tag(tmp_path):
    runner = _publish_runner(ancestry_returncode=1)

    with pytest.raises(ReleaseError, match="not contained in origin/main"):
        release_script.validate_publish("v1.2.3", root=tmp_path, runner=runner, event_name="workflow_dispatch")


def test_sha_target_release_payload_is_valid_when_tag_is_on_main(tmp_path):
    event = tmp_path / "event.json"
    event.write_text(
        '{"release":{"tag_name":"v1.2.3","target_commitish":"abc123","draft":false,'
        '"published_at":"2026-07-30T00:00:00Z"}}',
        encoding="utf-8",
    )
    runner = _publish_runner()

    assert release_script.validate_publish(
        "v1.2.3", root=tmp_path, runner=runner, event_name="release", event_path=event
    ) == ("1.2.3", "v1.2.3")


def test_build_uses_requested_version_override(tmp_path):
    _prepare_release(tmp_path)
    runner = _successful_runner()

    release_script.release("1.2.3", dry_run=True, root=tmp_path, runner=runner)

    build = next(item for item in runner.calls if item[0] == ("just", "build-check"))
    assert build[1] == {"SETUPTOOLS_SCM_PRETEND_VERSION": "1.2.3"}
