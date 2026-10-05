import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.dataset import load_mitbih


RAW_DIR = Path("data/raw/mitdb")
OUTPUT = Path("data/processed/mitbih_all.csv")

RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "106",
    "107",
    "108",
    "109",
]

MAX_BEATS = 2000
HISTORY = 10


def add_temporal_features(df, history=10):

    previous_rr = (
        df["rr_interval"]
        .shift(1)
    )

    rolling_rr_mean = (
        df["rr_interval"]
        .shift(1)
        .rolling(
            window=history,
            min_periods=1
        )
        .mean()
    )

    rolling_rr_std = (
        df["rr_interval"]
        .shift(1)
        .rolling(
            window=history,
            min_periods=2
        )
        .std()
    )

    rolling_hr_mean = (
        df["heart_rate"]
        .shift(1)
        .rolling(
            window=history,
            min_periods=1
        )
        .mean()
    )

    rolling_hr_std = (
        df["heart_rate"]
        .shift(1)
        .rolling(
            window=history,
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

    df.replace(
        [float("inf"), float("-inf")],
        float("nan"),
        inplace=True
    )

    df.fillna(0.0, inplace=True)

    return df


def main():

    print("=" * 70)
    print("BIO-TWIN 6G — MIT-BIH BATCH PREPROCESSING")
    print("=" * 70)

    datasets = []

    for record in RECORDS:

        record_path = RAW_DIR / record

        print(
            f"Processing record {record}..."
        )

        if not record_path.with_suffix(".hea").exists():

            print(
                f"  WARNING: record {record} not found"
            )

            continue

        df = load_mitbih(
            record_path,
            MAX_BEATS
        )

        if len(df) == 0:
            print(
                f"  WARNING: no valid beats"
            )
            continue

        # Important:
        # temporal history is calculated independently
        # for every patient/record.
        df = add_temporal_features(
            df,
            HISTORY
        )

        datasets.append(df)

        print(
            f"  Beats extracted: {len(df)}"
        )

    if not datasets:
        raise RuntimeError(
            "No MIT-BIH records were successfully processed."
        )

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("-" * 70)
    print(f"Records processed : {combined['record'].nunique()}")
    print(f"Total beats       : {len(combined)}")
    print(f"Output            : {OUTPUT}")

    print()
    print("Records:")
    print(
        combined["record"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Annotations:")
    print(
        combined["annotation"]
        .value_counts()
        .to_string()
    )

    print("=" * 70)


if __name__ == "__main__":
    main()