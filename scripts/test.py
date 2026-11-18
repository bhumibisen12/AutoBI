"""Helper script to run the AutoBI pytest suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    """Execute ``pytest`` from the repository root, forwarding extra CLI arguments."""
    extra_args = list(sys.argv[1:] if argv is None else argv)
    cmd = [sys.executable, "-m", "pytest", *extra_args]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

