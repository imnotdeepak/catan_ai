"""Build the frozen imitation dataset (ML-012) -> data/<config name>/.

Examples:
    python -m catan_ai.cli.imitation_data --workers 11
    python -m catan_ai.cli.imitation_data --source data/imitation-v3 --workers 11

With `--source`, the games are not played again: they are rebuilt from the
source dataset's replays and encoded with the current features (minutes,
not an hour). The output defaults to data/<name>-features-v<N>.
"""

import argparse
import json
import sys
from pathlib import Path

from catan_ai.models.encoding import LATEST_FEATURE_VERSION
from catan_ai.provenance import REPO_ROOT
from catan_ai.training.dataset import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "imitation.json")
    parser.add_argument("--out", type=Path, default=None, help="default data/<config name>")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source", type=Path, default=None, help="re-encode this dataset")
    parser.add_argument(
        "--feature-version", type=int, default=LATEST_FEATURE_VERSION, help="default latest"
    )
    args = parser.parse_args(argv)

    full = json.loads(args.config.read_text(encoding="utf-8"))
    full["dataset"]["feature_version"] = args.feature_version
    default = full["name"] + (f"-features-v{args.feature_version}" if args.source else "")
    out = args.out or REPO_ROOT / "data" / default
    if (out / "manifest.json").exists():
        print(f"{out} already holds a dataset; datasets are frozen", file=sys.stderr)
        return 1
    manifest = build_dataset(full["dataset"], out, args.workers, args.source)
    for name, split in manifest["splits"].items():
        print(
            f"{name}: {split['games']} games, {split['samples']:,} samples, "
            f"{split['candidates']:,} candidates, {split['minutes']} min"
        )
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
