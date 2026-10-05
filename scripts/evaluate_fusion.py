from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.neuro_symbolic import NeuroSymbolicIDS  # noqa: E402

# ============================================================
# FILE PATHS
# ============================================================

TEST_CSV = PROJECT / "data" / "processed" / "mitbih_test.csv"

MODEL_PATH = PROJECT / "models" / "lstm_detector_heldout.pt"

THRESHOLD_PATH = PROJECT / "models" / "lstm_threshold_heldout.npz"

FUSION_CONFIG_PATH = PROJECT / "models" / "neurosymbolic_fusion.json"

OUT_PATH = PROJECT / "results" / "neurosymbolic_fusion_heldout.csv"


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

SEQ_LEN = 20

TEST_RECORDS = [
    107,
    108,
    109,
]

FEATURES = [
    "heart_rate",
    "rr_interval",
    "ecg_mean",
    "ecg_std",
    "ecg_min",
    "ecg_max",
    "prev_rr",
    "rolling_rr_mean",
    "rolling_rr_std",
    "rr_deviation",
    "rolling_hr_mean",
    "rolling_hr_std",
    "hr_deviation",
    "rr_deviation_z",
    "hr_deviation_z",
]


# ============================================================
# LSTM AUTOENCODER
#
# EXACT architecture used during training.
# ============================================================


class LSTMAutoencoder(torch.nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
        latent_size=16,
    ):
        super().__init__()

        self.encoder = torch.nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
        )

        self.to_latent = torch.nn.Linear(
            hidden_size,
            latent_size,
        )

        self.from_latent = torch.nn.Linear(
            latent_size,
            hidden_size,
        )

        self.decoder = torch.nn.LSTM(
            hidden_size,
            input_size,
            batch_first=True,
        )

    def forward(self, x):

        encoded, _ = self.encoder(x)

        last_hidden = encoded[:, -1, :]

        latent = self.to_latent(last_hidden)

        decoder_input = self.from_latent(latent)

        decoder_input = decoder_input.unsqueeze(1)

        decoder_input = decoder_input.repeat(
            1,
            x.size(1),
            1,
        )

        reconstructed, _ = self.decoder(decoder_input)

        return reconstructed, latent


# ============================================================
# LOAD LSTM
# ============================================================


def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = LSTMAutoencoder(
        input_size=len(checkpoint["features"]),
        hidden_size=checkpoint["hidden_size"],
        latent_size=checkpoint["latent_size"],
    )

    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    return model, checkpoint


# ============================================================
# CREATE TEST SEQUENCES
# ============================================================


def make_sequences(
    df,
    mean,
    std,
):

    sequences = []
    rows = []

    for record, group in df.groupby(
        "record",
        sort=False,
    ):

        group = group.reset_index(drop=True)

        values = group[FEATURES].astype(float).values

        # Use ONLY the normalization statistics
        # stored in the training checkpoint.

        values = (values - mean) / np.where(
            std < 1e-8,
            1.0,
            std,
        )

        for end in range(
            SEQ_LEN - 1,
            len(group),
        ):

            start = end - SEQ_LEN + 1

            sequences.append(values[start:end + 1])

            rows.append(group.iloc[end])

    return (
        np.asarray(
            sequences,
            dtype=np.float32,
        ),
        pd.DataFrame(rows).reset_index(drop=True),
    )


# ============================================================
# METRICS
# ============================================================


def binary_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true).astype(int)

    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(
        tp + tn + fp + fn,
        1,
    )

    precision = tp / max(
        tp + fp,
        1,
    )

    recall = tp / max(
        tp + fn,
        1,
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    specificity = tn / max(
        tn + fp,
        1,
    )

    fpr = fp / max(
        fp + tn,
        1,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# EVALUATE ONE RECORD
# ============================================================


def evaluate_record(
    record_df,
    model,
    mean,
    std,
    threshold,
    neural_weight,
    symbolic_weight,
    decision_threshold,
    ids,
):

    # --------------------------------------------------------
    # Create sequences for this record
    # --------------------------------------------------------

    X, rows = make_sequences(
        record_df,
        mean,
        std,
    )

    if len(X) == 0:
        return None

    X_tensor = torch.from_numpy(X)

    # --------------------------------------------------------
    # LSTM reconstruction errors
    # --------------------------------------------------------

    errors = []

    with torch.no_grad():

        batch_size = 256

        for start in range(
            0,
            len(X_tensor),
            batch_size,
        ):

            batch = X_tensor[start:start + batch_size]

            reconstructed, _ = model(batch)

            mse = torch.mean(
                (reconstructed - batch) ** 2,
                dim=(1, 2),
            )

            errors.extend(mse.numpy().tolist())

    errors = np.asarray(
        errors,
        dtype=float,
    )

    # --------------------------------------------------------
    # Neural risk
    # --------------------------------------------------------

    neural_risk = np.minimum(
        errors / threshold,
        1.0,
    )

    neural_binary = (errors > threshold).astype(int)

    # --------------------------------------------------------
    # Symbolic detector
    # --------------------------------------------------------

    symbolic_binary = []
    flag_counts = []

    for _, row in rows.iterrows():

        features = {}

        for feature in FEATURES:

            value = row.get(feature)

            if pd.notna(value):

                features[feature] = float(value)

        detection = ids.evaluate(
            features,
            neural_score=0.0,
            fast_mode=True,
        )

        symbolic_binary.append(1 if detection.decision == "BLOCK" else 0)

        flag_counts.append(len(detection.flags))

    symbolic_binary = np.asarray(
        symbolic_binary,
        dtype=int,
    )

    flag_counts = np.asarray(
        flag_counts,
        dtype=float,
    )

    # --------------------------------------------------------
    # Symbolic risk
    # --------------------------------------------------------

    symbolic_risk = np.minimum(
        flag_counts / 2.0,
        1.0,
    )

    # --------------------------------------------------------
    # Frozen weighted fusion
    # --------------------------------------------------------

    fused_risk = neural_weight * neural_risk + symbolic_weight * symbolic_risk

    fusion_prediction = (fused_risk >= decision_threshold).astype(int)

    # --------------------------------------------------------
    # Ground truth
    #
    # N vs non-N physiological anomaly proxy.
    # NOT cyberattack ground truth.
    # --------------------------------------------------------

    y_true = (rows["annotation"].astype(str) != "N").astype(int).values

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = binary_metrics(
        y_true,
        fusion_prediction,
    )

    # Additional detector information
    metrics["record"] = int(record_df["record"].iloc[0])

    metrics["sequences"] = len(X)

    metrics["normal"] = int((y_true == 0).sum())

    metrics["non_normal"] = int((y_true == 1).sum())

    metrics["neural_anomalies"] = int(neural_binary.sum())

    metrics["symbolic_blocks"] = int(symbolic_binary.sum())

    metrics["fusion_blocks"] = int(fusion_prediction.sum())

    metrics["mean_reconstruction_error"] = float(errors.mean())

    metrics["median_reconstruction_error"] = float(np.median(errors))

    metrics["mean_neural_risk"] = float(neural_risk.mean())

    metrics["mean_symbolic_risk"] = float(symbolic_risk.mean())

    metrics["mean_fused_risk"] = float(fused_risk.mean())

    return metrics


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("NEURO-SYMBOLIC FUSION - HELD-OUT TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    required_files = [
        TEST_CSV,
        MODEL_PATH,
        THRESHOLD_PATH,
        FUSION_CONFIG_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(f"Required file not found:\n{path}")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading held-out test dataset...")

    df = pd.read_csv(TEST_CSV)

    print(f"Test rows: {len(df)}")

    available_records = sorted(df["record"].unique().tolist())

    print(
        "Available test records:",
        ", ".join(str(x) for x in available_records),
    )

    # Safety check:
    # only evaluate the predefined held-out records.

    missing_records = [r for r in TEST_RECORDS if r not in available_records]

    if missing_records:

        raise ValueError(
            "Missing expected test records: "
            + ", ".join(str(x) for x in missing_records)
        )

    # --------------------------------------------------------
    # Verify test records
    # --------------------------------------------------------

    test_df = df[df["record"].isin(TEST_RECORDS)].copy()

    print("\nRecords used for final evaluation:")

    for record in TEST_RECORDS:

        count = int((test_df["record"] == record).sum())

        print(f"  Record {record}: " f"{count} rows")

    # --------------------------------------------------------
    # Load frozen fusion configuration
    # --------------------------------------------------------

    print("\nLoading frozen fusion configuration...")

    with open(
        FUSION_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    print(
        "Calibration dataset:",
        config["calibration_dataset"],
    )

    print(
        "Calibration records:",
        ", ".join(str(x) for x in config["calibration_records"]),
    )

    print(
        "Fusion method:",
        config["method"],
    )

    neural_weight = float(config["neural_weight"])

    symbolic_weight = float(config["symbolic_weight"])

    decision_threshold = float(config["decision_threshold"])

    print(f"Neural weight:       " f"{neural_weight:.2f}")

    print(f"Symbolic weight:     " f"{symbolic_weight:.2f}")

    print(f"Decision threshold:  " f"{decision_threshold:.2f}")

    # --------------------------------------------------------
    # Safety verification
    # --------------------------------------------------------

    if set(TEST_RECORDS).intersection(set(config["calibration_records"])):

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            "a test record appears in "
            "the calibration records."
        )

    print("\nData-leakage check: PASSED")

    print("Test records were not used " "during fusion calibration.")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading frozen LSTM model...")

    model, checkpoint = load_model()

    print(
        "Model features:",
        len(checkpoint["features"]),
    )

    print(
        "Sequence length:",
        checkpoint["sequence_length"],
    )

    print(
        "Hidden size:",
        checkpoint["hidden_size"],
    )

    print(
        "Latent size:",
        checkpoint["latent_size"],
    )

    # --------------------------------------------------------
    # Load frozen threshold
    # --------------------------------------------------------

    threshold_data = np.load(THRESHOLD_PATH)

    threshold = float(threshold_data["threshold"])

    print(f"LSTM threshold: " f"{threshold:.6f}")

    # --------------------------------------------------------
    # Normalization statistics
    # --------------------------------------------------------

    mean = np.asarray(
        checkpoint["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["std"],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Frozen symbolic detector
    # --------------------------------------------------------

    ids = NeuroSymbolicIDS()

    # --------------------------------------------------------
    # Evaluate each held-out record
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("HELD-OUT RECORD RESULTS")

    print("=" * 70)

    results = []

    for record in TEST_RECORDS:

        print(f"\nEvaluating record {record}...")

        record_df = test_df[test_df["record"] == record].copy()

        result = evaluate_record(
            record_df=record_df,
            model=model,
            mean=mean,
            std=std,
            threshold=threshold,
            neural_weight=neural_weight,
            symbolic_weight=symbolic_weight,
            decision_threshold=decision_threshold,
            ids=ids,
        )

        if result is None:

            print(f"Record {record}: " "no valid sequences.")

            continue

        results.append(result)

        print(f"Sequences:    " f"{result['sequences']}")

        print(f"Normal:       " f"{result['normal']}")

        print(f"Non-N:        " f"{result['non_normal']}")

        print(f"Neural:       " f"{result['neural_anomalies']}")

        print(f"Symbolic:     " f"{result['symbolic_blocks']}")

        print(f"Fusion blocks: " f"{result['fusion_blocks']}")

        print(f"Accuracy:      " f"{result['accuracy']:.4f}")

        print(f"Precision:     " f"{result['precision']:.4f}")

        print(f"Recall:        " f"{result['recall']:.4f}")

        print(f"F1:            " f"{result['f1']:.4f}")

        print(f"Specificity:   " f"{result['specificity']:.4f}")

        print(f"FPR:           " f"{result['fpr']:.4f}")

    # --------------------------------------------------------
    # Overall held-out evaluation
    # --------------------------------------------------------

    if not results:

        raise RuntimeError("No held-out evaluation results were produced.")

    # Calculate overall predictions directly from
    # accumulated confusion-matrix values.

    total_tp = sum(r["tp"] for r in results)

    total_tn = sum(r["tn"] for r in results)

    total_fp = sum(r["fp"] for r in results)

    total_fn = sum(r["fn"] for r in results)

    total_samples = total_tp + total_tn + total_fp + total_fn

    overall_accuracy = (total_tp + total_tn) / max(
        total_samples,
        1,
    )

    overall_precision = total_tp / max(
        total_tp + total_fp,
        1,
    )

    overall_recall = total_tp / max(
        total_tp + total_fn,
        1,
    )

    overall_f1 = (
        2
        * overall_precision
        * overall_recall
        / max(
            overall_precision + overall_recall,
            1e-12,
        )
    )

    overall_specificity = total_tn / max(
        total_tn + total_fp,
        1,
    )

    overall_fpr = total_fp / max(
        total_fp + total_tn,
        1,
    )

    overall = {
        "record": "OVERALL",
        "sequences": total_samples,
        "normal": sum(r["normal"] for r in results),
        "non_normal": sum(r["non_normal"] for r in results),
        "neural_anomalies": sum(r["neural_anomalies"] for r in results),
        "symbolic_blocks": sum(r["symbolic_blocks"] for r in results),
        "fusion_blocks": sum(r["fusion_blocks"] for r in results),
        "accuracy": overall_accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "specificity": overall_specificity,
        "fpr": overall_fpr,
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "mean_reconstruction_error": np.mean(
            [r["mean_reconstruction_error"] for r in results]
        ),
        "median_reconstruction_error": np.mean(
            [r["median_reconstruction_error"] for r in results]
        ),
        "mean_neural_risk": np.mean([r["mean_neural_risk"] for r in results]),
        "mean_symbolic_risk": np.mean([r["mean_symbolic_risk"] for r in results]),  # noqa: E501
        "mean_fused_risk": np.mean([r["mean_fused_risk"] for r in results]),
    }

    # --------------------------------------------------------
    # Display overall result
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL HELD-OUT NEURO-SYMBOLIC RESULT")

    print("=" * 70)

    print("Records:       107, 108, 109")

    print(f"Sequences:     {overall['sequences']}")

    print(f"Accuracy:      {overall['accuracy']:.4f}")

    print(f"Precision:     {overall['precision']:.4f}")

    print(f"Recall:        {overall['recall']:.4f}")

    print(f"F1:            {overall['f1']:.4f}")

    print(f"Specificity:   {overall['specificity']:.4f}")

    print(f"FPR:           {overall['fpr']:.4f}")

    print(f"TP:            {overall['tp']}")

    print(f"TN:            {overall['tn']}")

    print(f"FP:            {overall['fp']}")

    print(f"FN:            {overall['fn']}")

    print(
        "\nMean reconstruction error:",
        f"{overall['mean_reconstruction_error']:.6f}",
    )

    print(
        "Mean neural risk:",
        f"{overall['mean_neural_risk']:.6f}",
    )

    print(
        "Mean symbolic risk:",
        f"{overall['mean_symbolic_risk']:.6f}",
    )

    print(
        "Mean fused risk:",
        f"{overall['mean_fused_risk']:.6f}",
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_rows = results + [overall]

    output_df = pd.DataFrame(output_rows)

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUT_PATH,
        index=False,
    )

    print("\nSaved held-out results to:")

    print(OUT_PATH)

    print("\nHeld-out evaluation complete.")

    print("No test-time calibration or threshold tuning was performed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
