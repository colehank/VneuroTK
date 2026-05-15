# /// script
# requires-python = ">=3.11"
# ///
"""Tag the current version and create a GitHub release.

Usage: uv run scripts/release.py <version>
Example: uv run scripts/release.py 1.0.0
"""

import subprocess
import sys
from pathlib import Path


def _run(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}")  # noqa: T201
    subprocess.run(cmd, check=True)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/release.py <version>")  # noqa: T201
        print("Example: uv run scripts/release.py 1.0.0")  # noqa: T201
        sys.exit(1)

    version = sys.argv[1].lstrip("v")
    tag = f"v{version}"
    notes_path = Path(f"CHANGELOG/{version}.md")

    if not notes_path.exists():
        print(f"Error: changelog not found at {notes_path}")  # noqa: T201
        sys.exit(1)

    lines = notes_path.read_text().splitlines(keepends=True)
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    notes = "".join(lines).rstrip()

    _run("git", "tag", "-a", tag, "-m", f"Release {tag}")
    _run("git", "push", "origin", tag)
    _run(
        "gh",
        "release",
        "create",
        tag,
        "--verify-tag",
        "--title",
        f"vneurotk {version}",
        "--notes",
        notes,
    )


if __name__ == "__main__":
    main()
