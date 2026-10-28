"""Build the frozen imitation dataset (ML-012) -> data/<config name>/.

Example:
    python -m catan_ai.cli.imitation_data --workers 11
"""

import argparse
import json
import sys
from pathlib import Path

from catan_ai.provenance import REPO_ROOT
from catan_ai.training.dataset import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "imitation.json")
    parser.add_argument("--out", type=Path, default=None, help="default data/<config name>")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    full = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.out or REPO_ROOT / "data" / full["name"]
    if (out / "manifest.json").exists():
        print(f"{out} already holds a dataset; datasets are frozen", file=sys.stderr)
        return 1
    manifest = build_dataset(full["dataset"], out, args.workers)
    for name, split in manifest["splits"].items():
        print(
            f"{name}: {split['games']} games, {split['samples']:,} samples, "
            f"{split['candidates']:,} candidates, {split['minutes']} min"
        )
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
