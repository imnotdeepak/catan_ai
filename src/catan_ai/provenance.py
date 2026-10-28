"""Versions recorded with every result so any run can be reproduced (plan §11.1)."""

import subprocess
from functools import cache
from pathlib import Path

import catan_ai

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


@cache
def provenance() -> dict[str, str | bool]:
    return {
        "catan_ai_version": catan_ai.__version__,
        "catan_ai_commit": _git("rev-parse", "HEAD", cwd=REPO_ROOT),
        "catan_ai_dirty": bool(_git("status", "--porcelain", "--", "src", cwd=REPO_ROOT)),
        "engine_commit": _git("rev-parse", "HEAD", cwd=REPO_ROOT / "vendor" / "catanatron"),
    }
