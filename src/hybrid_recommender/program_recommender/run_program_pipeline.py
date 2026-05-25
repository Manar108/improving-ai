from __future__ import annotations

"""CLI entrypoint for the program recommender pipeline."""

import argparse
from pathlib import Path

from .pipeline import run_program_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the program recommender pipeline")
    parser.add_argument(
        "--raw-data",
        type=str,
        default=None,
        help="Optional path to a folder containing normalized DB CSVs (e.g. programs.csv). If omitted, the pipeline loads from SQL Server via the existing preprocessing layer.",
    )
    args = parser.parse_args()

    raw_data = Path(args.raw_data) if args.raw_data else None
    artifacts = run_program_pipeline(raw_data=raw_data)

    metrics = {
        "valid": artifacts.get("metrics_valid", {}),
        "test_raw": artifacts.get("metrics_test_raw", {}),
        "test_reranked": artifacts.get("metrics_test_reranked", {}),
    }
    print("Program pipeline finished successfully.")
    for name, metric_block in metrics.items():
        print(f"[{name}]")
        for key, value in metric_block.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
