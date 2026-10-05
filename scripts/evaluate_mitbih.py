import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.pipeline import BioTwin6GPipeline


DATASET = Path(
    "data/processed/mitbih_all.csv"
)


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    tp = sum(
        t == "non-N" and p == "non-N"
        for t, p in zip(y_true, y_pred)
    )

    tn = sum(
        t == "N" and p == "N"
        for t, p in zip(y_true, y_pred)
    )

    fp = sum(
        t == "N" and p == "non-N"
        for t, p in zip(y_true, y_pred)
    )

    fn = sum(
        t == "non-N" and p == "N"
        for t, p in zip(y_true, y_pred)
    )

    total = len(y_true)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
    }


def evaluate_record(
    df,
    pipeline,
    rule_counts
):

    y_true = []
    y_pred = []
    latencies = []

    for _, row in df.iterrows():

        result = pipeline.process_mitbih_row(row)

        # -------------------------------------------------------------
        # Ground truth
        #
        # N     = reference normal beat
        # others = non-N annotation
        #
        # We deliberately call this "non-N", rather than claiming
        # every non-N MIT-BIH annotation is clinically abnormal.
        # -------------------------------------------------------------

        true_label = (
            "N"
            if row["annotation"] == "N"
            else "non-N"
        )

        # -------------------------------------------------------------
        # Security decision
        #
        # ALLOW = normal prediction
        # BLOCK = non-N prediction
        # -------------------------------------------------------------

        predicted_label = (
            "N"
            if result.ids_decision == "ALLOW"
            else "non-N"
        )

        y_true.append(true_label)
        y_pred.append(predicted_label)

        latencies.append(
            result.latency_ms
        )

        # -------------------------------------------------------------
        # Capture symbolic rule activations
        # -------------------------------------------------------------

        detection = pipeline.ids.evaluate(
            {
                "heart_rate": row["heart_rate"],
                "rr_interval": row["rr_interval"],
                "ecg_mean": row["ecg_mean"],
                "ecg_std": row["ecg_std"],
                "ecg_min": row["ecg_min"],
                "ecg_max": row["ecg_max"],
                "rr_deviation_z": row["rr_deviation_z"],
                "hr_deviation_z": row["hr_deviation_z"],
            }
        )

        for flag in detection.flags:

            rule_counts[flag] = (
                rule_counts.get(flag, 0) + 1
            )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    latency_series = pd.Series(
        latencies,
        dtype=float
    )

    metrics["count"] = len(df)
    metrics["n_count"] = y_true.count("N")
    metrics["non_n_count"] = y_true.count("non-N")

    metrics["mean_latency"] = latency_series.mean()
    metrics["median_latency"] = latency_series.median()
    metrics["min_latency"] = latency_series.min()
    metrics["max_latency"] = latency_series.max()

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("BIO-TWIN 6G — MIT-BIH 9-RECORD EVALUATION")
    print("=" * 70)

    if not DATASET.exists():

        raise FileNotFoundError(
            f"Combined dataset not found: {DATASET}"
        )

    # -------------------------------------------------------------
    # Reproducibility
    #
    # The pipeline includes Gaussian DP noise. Fixing NumPy's seed
    # makes the evaluation repeatable.
    # -------------------------------------------------------------

    np.random.seed(42)

    df = pd.read_csv(DATASET)

    required_columns = [
        "record",
        "annotation",
        "heart_rate",
        "rr_interval",
        "ecg_mean",
        "ecg_std",
        "ecg_min",
        "ecg_max",
        "rr_deviation_z",
        "hr_deviation_z",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # -------------------------------------------------------------
    # Fixed pipeline
    #
    # The same IDS rules are used for every record.
    # -------------------------------------------------------------

    pipeline = BioTwin6GPipeline()

    all_true = []
    all_pred = []
    all_latencies = []

    overall_rule_counts = {}

    per_record = []

    # -------------------------------------------------------------
    # Evaluate every record independently
    # -------------------------------------------------------------

    records = sorted(
        df["record"].astype(str).unique()
    )

    print()
    print("DATASET")
    print("-" * 70)
    print(
        f"Records        : {len(records)}"
    )
    print(
        f"Total beats    : {len(df)}"
    )

    print()
    print("ANNOTATIONS")
    print("-" * 70)

    annotation_counts = (
        df["annotation"]
        .value_counts()
    )

    for annotation, count in annotation_counts.items():

        print(
            f"{str(annotation):<10} {count}"
        )

    # -------------------------------------------------------------
    # Record-level evaluation
    # -------------------------------------------------------------

    for record in records:

        record_df = df[
            df["record"].astype(str) == record
        ].copy()

        record_rules = {}

        # Capture labels/predictions separately for aggregation
        record_true = []
        record_pred = []
        record_latencies = []

        for _, row in record_df.iterrows():

            result = pipeline.process_mitbih_row(row)

            true_label = (
                "N"
                if row["annotation"] == "N"
                else "non-N"
            )

            predicted_label = (
                "N"
                if result.ids_decision == "ALLOW"
                else "non-N"
            )

            record_true.append(true_label)
            record_pred.append(predicted_label)
            record_latencies.append(
                result.latency_ms
            )

            detection = pipeline.ids.evaluate(
                {
                    "heart_rate": row["heart_rate"],
                    "rr_interval": row["rr_interval"],
                    "ecg_mean": row["ecg_mean"],
                    "ecg_std": row["ecg_std"],
                    "ecg_min": row["ecg_min"],
                    "ecg_max": row["ecg_max"],
                    "rr_deviation_z": row["rr_deviation_z"],
                    "hr_deviation_z": row["hr_deviation_z"],
                }
            )

            for flag in detection.flags:

                record_rules[flag] = (
                    record_rules.get(flag, 0) + 1
                )

                overall_rule_counts[flag] = (
                    overall_rule_counts.get(flag, 0) + 1
                )

        metrics = calculate_metrics(
            record_true,
            record_pred
        )

        latency_series = pd.Series(
            record_latencies,
            dtype=float
        )

        metrics["record"] = record
        metrics["count"] = len(record_df)
        metrics["n_count"] = record_true.count("N")
        metrics["non_n_count"] = record_true.count("non-N")
        metrics["mean_latency"] = latency_series.mean()
        metrics["median_latency"] = latency_series.median()
        metrics["min_latency"] = latency_series.min()
        metrics["max_latency"] = latency_series.max()

        per_record.append(metrics)

        # Aggregate
        all_true.extend(record_true)
        all_pred.extend(record_pred)
        all_latencies.extend(record_latencies)

    # -------------------------------------------------------------
    # Overall results
    # -------------------------------------------------------------

    overall = calculate_metrics(
        all_true,
        all_pred
    )

    latency_series = pd.Series(
        all_latencies,
        dtype=float
    )

    # -------------------------------------------------------------
    # Print per-record results
    # -------------------------------------------------------------

    print()
    print("PER-RECORD RESULTS")
    print("-" * 70)

    print(
        f"{'Record':<8}"
        f"{'Beats':>8}"
        f"{'N':>8}"
        f"{'non-N':>8}"
        f"{'Acc':>10}"
        f"{'Prec':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    for m in per_record:

        print(
            f"{m['record']:<8}"
            f"{m['count']:>8}"
            f"{m['n_count']:>8}"
            f"{m['non_n_count']:>8}"
            f"{m['accuracy']:>10.4f}"
            f"{m['precision']:>10.4f}"
            f"{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}"
        )

    # -------------------------------------------------------------
    # Overall ground truth
    # -------------------------------------------------------------

    print()
    print("OVERALL GROUND TRUTH")
    print("-" * 70)

    print(
        f"N beats       : "
        f"{all_true.count('N')}"
    )

    print(
        f"Non-N beats   : "
        f"{all_true.count('non-N')}"
    )

    # -------------------------------------------------------------
    # Overall confusion matrix
    # -------------------------------------------------------------

    print()
    print("OVERALL CONFUSION MATRIX")
    print("-" * 70)

    print(
        f"True Positive  : {overall['tp']}"
    )

    print(
        f"True Negative  : {overall['tn']}"
    )

    print(
        f"False Positive : {overall['fp']}"
    )

    print(
        f"False Negative : {overall['fn']}"
    )

    # -------------------------------------------------------------
    # Overall classification metrics
    # -------------------------------------------------------------

    print()
    print("OVERALL CLASSIFICATION METRICS")
    print("-" * 70)

    print(
        f"Accuracy       : "
        f"{overall['accuracy']:.4f}"
    )

    print(
        f"Precision      : "
        f"{overall['precision']:.4f}"
    )

    print(
        f"Recall         : "
        f"{overall['recall']:.4f}"
    )

    print(
        f"F1-score       : "
        f"{overall['f1']:.4f}"
    )

    print(
        f"Specificity    : "
        f"{overall['specificity']:.4f}"
    )

    print(
        f"False positive : "
        f"{overall['fpr']:.4f}"
    )

    # -------------------------------------------------------------
    # Latency
    # -------------------------------------------------------------

    print()
    print("OVERALL LATENCY")
    print("-" * 70)

    print(
        f"Mean           : "
        f"{latency_series.mean():.3f} ms"
    )

    print(
        f"Median         : "
        f"{latency_series.median():.3f} ms"
    )

    print(
        f"Minimum        : "
        f"{latency_series.min():.3f} ms"
    )

    print(
        f"Maximum        : "
        f"{latency_series.max():.3f} ms"
    )

    # -------------------------------------------------------------
    # Rule activations
    # -------------------------------------------------------------

    print()
    print("ECG SYMBOLIC RULE ACTIVATIONS")
    print("-" * 70)

    for rule, count in sorted(
        overall_rule_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{rule:<45} {count}"
        )

    # -------------------------------------------------------------
    # Save per-record results
    # -------------------------------------------------------------

    results_df = pd.DataFrame(
        per_record
    )

    output_path = Path(
        "results/mitbih_9record_evaluation.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("-" * 70)
    print(
        f"Per-record results saved to:"
    )
    print(
        f"{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()