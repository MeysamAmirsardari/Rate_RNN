"""Command-line entry point for the Nutmeg AB/BA analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import COMPARISONS, DEFAULT_EXPORT_FILE, DEFAULT_SOURCE_DIR
from .data import load_export
from .decoder import run_decoder
from .open_ephys import inventory_archive
from .reference import reference_manifest


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the leakage-audited Nutmeg AB/BA ECoG decoder."
    )
    parser.add_argument(
        "analysis", choices=[*COMPARISONS, "all", "inventory", "reference"]
    )
    parser.add_argument(
        "--mode",
        choices=["matlab-faithful", "leakage-safe", "both"],
        default="leakage-safe",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--export-file", type=Path, default=DEFAULT_EXPORT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seed", type=int,
        help="Declared audit seed. The source MATLAB figure RNG state is unknown.",
    )
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")


def _legacy_reference_check(
    reference: dict, key: str, result
) -> dict:
    expected = reference.get("reference_figures", {}).get(key)
    if not expected:
        return {"comparison": key, "reference_available": False}
    observed_source_label = int(result.source_time_labels_ms[result.peak_index])
    observed_channels = result.top_channels_matlab.tolist()
    expected_peak = expected["legacy_peak_time_source_label_ms"]
    expected_channels = expected["legacy_top_channels_matlab"]
    return {
        "comparison": key,
        "reference_available": True,
        "reference_file": expected["file"],
        "reference_sha256": expected["sha256"],
        "expected_peak_source_label_ms": expected_peak,
        "observed_peak_source_label_ms": observed_source_label,
        "peak_matches": observed_source_label == expected_peak,
        "expected_top_channels_matlab": expected_channels,
        "observed_top_channels_matlab": observed_channels,
        "top_channels_match": observed_channels == expected_channels,
        "exact_match_claimed": False,
        "reason": (
            "The supplied source script did not seed randn/KFold, so the saved "
            "figure's RNG state is not recoverable. This check reports observed "
            "agreement without treating disagreement as a translation failure."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.analysis == "inventory":
        destination = args.output_dir / "raw_inventory.json"
        _write_json(destination, inventory_archive(args.source_dir))
        print(f"Validated raw Open Ephys archive; saved {destination}")
        return 0
    if args.analysis == "reference":
        destination = args.output_dir / "reference_figures.json"
        _write_json(destination, reference_manifest(args.source_dir))
        print(f"Inventoried supplied MATLAB figures; saved {destination}")
        return 0

    keys = list(COMPARISONS) if args.analysis == "all" else [args.analysis]
    reference = reference_manifest(args.source_dir)
    modes = (
        ["matlab-faithful", "leakage-safe"] if args.mode == "both" else [args.mode]
    )
    failures = []
    for key in keys:
        spec = COMPARISONS[key]
        try:
            print(f"[{key}] loading {args.export_file}")
            epochs = load_export(args.export_file, spec)
            # Import plotting only after the scientific input has passed all
            # schema checks, so a missing export fails cleanly and quickly.
            from .outputs import save_analysis
            for mode in modes:
                print(f"[{key}] decoding mode={mode}")
                result = run_decoder(epochs, spec, mode=mode, random_seed=args.seed)
                destination = save_analysis(
                    args.output_dir, spec, args.export_file, epochs, result
                )
                if mode == "matlab-faithful":
                    _write_json(
                        destination / "legacy_reference_check.json",
                        _legacy_reference_check(reference, key, result),
                    )
                print(
                    f"[{key}] descriptive peak={result.peak_time_ms} ms, "
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
