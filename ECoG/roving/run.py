"""Command-line entry point for the roving ECoG decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ANALYSES, DEFAULT_SOURCE_DIR
from .decoder import run_decoder
from .matlab_io import extract_roving_epochs
from .outputs import save_analysis


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate and run the requested MATLAB roving decoder."
    )
    parser.add_argument("analysis", choices=[*ANALYSES, "available", "all"])
    parser.add_argument(
        "--mode",
        choices=["matlab-faithful", "leakage-safe", "both"],
        default="leakage-safe",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Explicit recording override for one analysis (never inferred).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--safe-spatial-window-ms",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        default=(0, 180),
        help="Prespecified half-open window relative to deviant onset.",
    )
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
    if args.data_file is not None and len(keys) != 1:
        raise SystemExit("--data-file can only be used with one named analysis")

    modes = (
        ["matlab-faithful", "leakage-safe"]
        if args.mode == "both"
        else [args.mode]
    )
    failures = []
    for key in keys:
        spec = ANALYSES[key]
        data_path = (
            args.data_file
            if args.data_file is not None
            else spec.data_path(args.source_dir)
        )
        try:
            print(f"[{key}] extracting {data_path}")
            epochs = extract_roving_epochs(data_path, spec)
            for mode in modes:
                print(f"[{key}] decoding mode={mode}")
                result = run_decoder(
                    epochs,
                    spec,
                    mode=mode,
                    safe_spatial_window_deviant_ms=tuple(
                        args.safe_spatial_window_ms
                    ),
                )
                destination = save_analysis(
                    args.output_dir, spec, data_path, epochs, result
                )
                print(
                    f"[{key}] peak={result.peak_time_ms} ms, "
                    f"smoothed accuracy={result.peak_accuracy_smoothed:.3f}; "
                    f"saved {destination}"
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

