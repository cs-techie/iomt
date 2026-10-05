import argparse
from pathlib import Path

import pandas as pd


TRAIN_RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "106",
]

TEST_RECORDS = [
    "107",
    "108",
    "109",
]


def main():

    parser = argparse.ArgumentParser(
        description="Create record-level MIT-BIH train/test split"
    )

    parser.add_argument(
        "--input",
        default="data/processed/mitbih_all.csv"
    )

    parser.add_argument(
        "--train-out",
        default="data/processed/mitbih_train.csv"
    )

    parser.add_argument(
        "--test-out",
        default="data/processed/mitbih_test.csv"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    df["record"] = (
        df["record"]
        .astype(str)
    )

    available = set(
        df["record"].unique()
    )

    missing_train = [
        r for r in TRAIN_RECORDS
        if r not in available
    ]

    missing_test = [
        r for r in TEST_RECORDS
        if r not in available
    ]

    if missing_train:
        raise ValueError(
            "Missing training records: "
            + ", ".join(missing_train)
        )

    if missing_test:
        raise ValueError(
            "Missing test records: "
            + ", ".join(missing_test)
        )

    overlap = (
        set(TRAIN_RECORDS)
        & set(TEST_RECORDS)
    )

    if overlap:
        raise ValueError(
            "Train/test overlap detected: "
            + ", ".join(sorted(overlap))
        )

    train_df = df[
        df["record"].isin(TRAIN_RECORDS)
    ].copy()

    test_df = df[
        df["record"].isin(TEST_RECORDS)
    ].copy()

    Path(args.train_out).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(args.test_out).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        args.train_out,
        index=False
    )

    test_df.to_csv(
        args.test_out,
        index=False
    )

    print("=" * 70)
    print(
        "BIO-TWIN 6G — RECORD-LEVEL DATASET SPLIT"
    )
    print("=" * 70)

    print()
    print("TRAINING RECORDS")
    print("-" * 70)

    for record in TRAIN_RECORDS:
        count = len(
            train_df[
                train_df["record"] == record
            ]
        )

        print(
            f"{record:<8} {count:>8} beats"
        )

    print(
        f"Total training beats: "
        f"{len(train_df)}"
    )

    print()
    print("HELD-OUT TEST RECORDS")
    print("-" * 70)

    for record in TEST_RECORDS:
        count = len(
            test_df[
                test_df["record"] == record
            ]
        )

        print(
            f"{record:<8} {count:>8} beats"
        )

    print(
        f"Total test beats: "
        f"{len(test_df)}"
    )

    print()
    print("ANNOTATION DISTRIBUTION — TRAIN")
    print("-" * 70)

    print(
        train_df["annotation"]
        .value_counts()
        .to_string()
    )

    print()
    print("ANNOTATION DISTRIBUTION — TEST")
    print("-" * 70)

    print(
        test_df["annotation"]
        .value_counts()
        .to_string()
    )

    print()
    print("-" * 70)

    print(
        f"Training file: {args.train_out}"
    )

    print(
        f"Test file    : {args.test_out}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()