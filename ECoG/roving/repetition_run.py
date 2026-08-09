"""Command-line runner for the ridge-logistic repetition posterior map."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ANALYSES, DEFAULT_SOURCE_DIR
from .matlab_io import extract_repetition_epochs
from .repetition_map import RepetitionMapConfig, run_repetition_map
from .repetition_outputs import save_repetition_map


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate SVM_rep_map.m with ridge logistic regression and save "
            "trial-level Rep-1-like posterior maps."
        )
    )
    parser.add_argument("analysis", choices=[*ANALYSES, "available", "all"])
    parser.add_argument(
        "--mode",
        choices=["matlab-faithful", "leakage-safe", "both"],
        default="leakage-safe",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.analysis == "available":
        keys = [
            key
            for key, spec in ANALYSES.items()
            if spec.data_path(args.source_dir).exists()
        ]
    elif args.analysis == "all":
        keys = list(ANALYSES)
    else:
        keys = [args.analysis]
    modes = (
        ["matlab-faithful", "leakage-safe"]
        if args.mode == "both"
        else [args.mode]
    )
    config = RepetitionMapConfig(
        ridge_lambda=args.ridge_lambda,
        n_folds=args.folds,
        random_seed=args.seed,
    )

    failures = []
    for key in keys:
        spec = ANALYSES[key]
        data_path = spec.data_path(args.source_dir)
        try:
            print(f"[{key}] extracting all 15 repetitions from {data_path}")
            epochs = extract_repetition_epochs(data_path, spec)
            for mode in modes:
                print(f"[{key}] ridge-logistic posterior mode={mode}")
                result = run_repetition_map(
                    epochs, spec, mode=mode, config=config
                )
                destination = save_repetition_map(
                    args.output_dir,
                    args.source_dir,
                    data_path,
                    spec,
                    epochs,
                    result,
                    config,
                )
                print(
                    f"[{key}] mean endpoint AUC="
                    f"{result.endpoint_auc.mean():.3f}; saved {destination}"
                )
        except Exception as error:
            failures.append((key, error))
            print(f"[{key}] FAILED: {error}")

    if failures:
        print("\nFailures:")
        for key, error in failures:
            print(f"  {key}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
