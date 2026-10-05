import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.dataset import load_mitbih


parser = argparse.ArgumentParser(
    description="Prepare MIT-BIH ECG features"
)

parser.add_argument(
    "--record",
    required=True,
    help="MIT-BIH record name, e.g. 100"
)

parser.add_argument(
    "--beats",
    type=int,
    default=2000,
    help="Maximum number of beats to extract"
)

parser.add_argument(
    "--history",
    type=int,
    default=10,
    help="Number of previous beats used for temporal baseline"
)

parser.add_argument(
    "--raw-dir",
    default="data/raw/mitdb",
    help="Directory containing MIT-BIH records"
)

parser.add_argument(
    "--out",
    default=None,
    help="Output CSV path"
)

args = parser.parse_args()

record_path = Path(args.raw_dir) / args.record

if not (record_path.with_suffix(".hea")).exists():
    raise FileNotFoundError(
        f"MIT-BIH record not found: {record_path}\n"
        f"Expected files such as:\n"
        f"  {record_path}.hea\n"
        f"  {record_path}.dat\n"
        f"  {record_path}.atr"
    )

df = load_mitbih(
    record_path,
    args.beats
)

if len(df) > 0:

    # ---------------------------------------------------------
    # Temporal ECG baseline
    #
    # IMPORTANT:
    # shift(1) ensures the current beat is NOT included in
    # its own baseline. This avoids information leakage.
    # ---------------------------------------------------------

    previous_rr = (
        df["rr_interval"]
        .shift(1)
    )

    previous_hr = (
        df["heart_rate"]
        .shift(1)
    )

    rolling_rr_mean = (
        df["rr_interval"]
        .shift(1)
        .rolling(
            window=args.history,
            min_periods=1
        )
        .mean()
    )

    rolling_rr_std = (
        df["rr_interval"]
        .shift(1)
        .rolling(
            window=args.history,
            min_periods=2
        )
        .std()
    )

    rolling_hr_mean = (
        df["heart_rate"]
        .shift(1)
        .rolling(
            window=args.history,
            min_periods=1
        )
        .mean()
    )

    rolling_hr_std = (
        df["heart_rate"]
        .shift(1)
        .rolling(
            window=args.history,
            min_periods=2
        )
        .std()
    )

    df["prev_rr"] = previous_rr

    df["rolling_rr_mean"] = (
        rolling_rr_mean
        .fillna(df["rr_interval"])
    )

    df["rolling_rr_std"] = (
        rolling_rr_std
        .fillna(0.0)
    )

    df["rr_deviation"] = (
        df["rr_interval"]
        - df["rolling_rr_mean"]
    )

    df["rolling_hr_mean"] = (
        rolling_hr_mean
        .fillna(df["heart_rate"])
    )

    df["rolling_hr_std"] = (
        rolling_hr_std
        .fillna(0.0)
    )

    df["hr_deviation"] = (
        df["heart_rate"]
        - df["rolling_hr_mean"]
    )

    # Normalized RR deviation.
    # Small denominator protection prevents division by zero.
    df["rr_deviation_z"] = (
        df["rr_deviation"]
        /
        df["rolling_rr_std"].clip(lower=0.01)
    )

    df["hr_deviation_z"] = (
        df["hr_deviation"]
        /
        df["rolling_hr_std"].clip(lower=0.5)
    )

    # Replace any numerical infinities created by very small
    # standard deviations.
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.fillna(0.0, inplace=True)


output = Path(
    args.out
    or f"data/processed/mitbih_{args.record}.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output,
    index=False
)

print("=" * 60)
print("MIT-BIH PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Record       : {args.record}")
print(f"Beats        : {len(df)}")
print(f"History      : {args.history} previous beats")
print(f"Output       : {output}")

print()

if len(df) > 0:

    print("Features:")

    for column in df.columns:
        print(f"  - {column}")

    print()

    print("First 5 rows:")

    print(
        df.head().to_string(
            index=False
        )
    )

print("=" * 60)