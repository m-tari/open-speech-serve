from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from bench.harness import load_cell_config, run_cell


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sweep of cell configs")
    parser.add_argument("sweep", help="Path to sweep YAML")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with Path(args.sweep).open() as f:
        sweep = yaml.safe_load(f)

    cells = sweep.get("cells") or []
    base = Path(args.sweep).parent
    for cell_path in cells:
        path = Path(cell_path)
        if not path.is_absolute():
            # Resolve relative to repo root first, then sweep dir.
            candidates = [Path(cell_path), base / cell_path, Path("configs") / cell_path]
            path = next((c for c in candidates if c.exists()), Path(cell_path))
        print(f"=== {path} ===")
        cfg = load_cell_config(path)
        run_cell(cfg, Path(args.results_dir))


if __name__ == "__main__":
    main()
