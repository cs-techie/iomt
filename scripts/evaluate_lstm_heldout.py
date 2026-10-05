import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)


MODEL_PATH = Path(
    "models/lstm_detector_heldout.pt"
)

THRESHOLD_PATH = Path(
    "models/lstm_threshold_heldout.npz"
)

DATASET_PATH = Path(
    "data/processed/mitbih_test.csv"
)

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
        latent_size=16
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )

        self.to_latent = nn.Linear(
            hidden_size,
            latent_size
        )

        self.from_latent = nn.Linear(
            latent_size,
            hidden_size
        )

        self.decoder = nn.LSTM(
            hidden_size,
            input_size,
            batch_first=True
        )

    def forward(self, x):

        encoded, _ = self.encoder(x)

        last_hidden = encoded[:, -1, :]

        latent = self.to_latent(
            last_hidden
        )

        decoder_input = self.from_latent(
            latent
        )

        decoder_input = decoder_input.unsqueeze(1)

        decoder_input = decoder_input.repeat(
            1,
            x.size(1),
            1
        )

        reconstructed, _ = self.decoder(
            decoder_input
        )

        return reconstructed, latent


def calculate_metrics(
    y_true,
    y_pred
):

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
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
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


def create_sequences(
    record_df,
    features,
    mean,
    std,
    sequence_length
):

    values = (
        record_df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )

    values = (
        values - mean
    ) / std

    sequences = []
    labels = []
    indices = []

    for i in range(
        len(values) - sequence_length + 1
    ):

        sequences.append(
            values[
                i:i + sequence_length
            ]
        )

        # The prediction belongs to the final
        # beat in the sequence.
        labels.append(
            "N"
            if record_df.iloc[
                i + sequence_length - 1
            ]["annotation"] == "N"
            else "non-N"
        )

        indices.append(
            i + sequence_length - 1
        )

    if not sequences:

        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    len(features)
                ),
                dtype=np.float32
            ),
            [],
            []
        )

    return (
        np.stack(
            sequences
        ).astype(np.float32),
        labels,
        indices
    )


def main():

    print("=" * 70)
    print(
    "BIO-TWIN 6G — HELD-OUT LSTM EVALUATION"
     )
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold not found: {THRESHOLD_PATH}"
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    threshold_data = np.load(
        THRESHOLD_PATH
    )

    threshold = float(
        threshold_data["threshold"]
    )

    features = checkpoint["features"]

    mean = np.asarray(
        checkpoint["mean"],
        dtype=np.float32
    )

    std = np.asarray(
        checkpoint["std"],
        dtype=np.float32
    )

    sequence_length = int(
        checkpoint["sequence_length"]
    )

    model = LSTMAutoencoder(
        input_size=len(features),
        hidden_size=int(
            checkpoint.get(
                "hidden_size",
                32
            )
        ),
        latent_size=int(
            checkpoint.get(
                "latent_size",
                16
            )
        )
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    df = pd.read_csv(
        DATASET_PATH
    )

    all_true = []
    all_pred = []
    all_errors = []

    per_record = []

    print()
    print(
        f"Threshold        : {threshold:.6f}"
    )

    print(
        f"Sequence length  : {sequence_length}"
    )

    print(
        f"Records           : "
        f"{df['record'].nunique()}"
    )

    # -------------------------------------------------------------
    # Evaluate each record separately
    # -------------------------------------------------------------

    for record in sorted(
        df["record"].astype(str).unique()
    ):

        record_df = df[
            df["record"].astype(str) == record
        ].reset_index(
            drop=True
        )

        X, labels, indices = create_sequences(
            record_df,
            features,
            mean,
            std,
            sequence_length
        )

        if len(X) == 0:
            continue

        tensor_x = torch.tensor(
            X,
            dtype=torch.float32
        )

        errors = []

        with torch.no_grad():

            for start in range(
                0,
                len(tensor_x),
                256
            ):

                batch = tensor_x[
                    start:start + 256
                ]

                reconstructed, _ = model(
                    batch
                )

                mse = (
                    (reconstructed - batch) ** 2
                ).mean(
                    dim=(1, 2)
                )

                errors.extend(
                    mse.numpy().tolist()
                )

        predictions = [
            "non-N"
            if error > threshold
            else "N"
            for error in errors
        ]

        metrics = calculate_metrics(
            labels,
            predictions
        )

        metrics["record"] = record
        metrics["sequences"] = len(labels)

        per_record.append(
            metrics
        )

        all_true.extend(labels)
        all_pred.extend(predictions)
        all_errors.extend(errors)

    # -------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------

    overall = calculate_metrics(
        all_true,
        all_pred
    )

    print()
    print("PER-RECORD RESULTS")
    print("-" * 70)

    print(
        f"{'Record':<8}"
        f"{'Seq':>8}"
        f"{'Acc':>10}"
        f"{'Prec':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    for m in per_record:

        print(
            f"{m['record']:<8}"
            f"{m['sequences']:>8}"
            f"{m['accuracy']:>10.4f}"
            f"{m['precision']:>10.4f}"
            f"{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}"
        )

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

    errors = np.asarray(
        all_errors
    )

    print()
    print("RECONSTRUCTION ERROR")
    print("-" * 70)

    print(
        f"Mean           : "
        f"{errors.mean():.6f}"
    )

    print(
        f"Median         : "
        f"{np.median(errors):.6f}"
    )

    print(
        f"95th percentile: "
        f"{np.percentile(errors, 95):.6f}"
    )

    print(
        f"99th percentile: "
        f"{np.percentile(errors, 99):.6f}"
    )

    print(
        f"Maximum        : "
        f"{errors.max():.6f}"
    )

    output = Path(
        "results/lstm_9record_evaluation.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(
        per_record
    ).to_csv(
        output,
        index=False
    )

    print()
    print(
        f"Results saved to: {output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()