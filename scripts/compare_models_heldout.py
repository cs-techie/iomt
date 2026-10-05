from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.neuro_symbolic import NeuroSymbolicIDS


# ============================================================
# PATHS
# ============================================================

TEST_CSV = (
    PROJECT
    / "data"
    / "processed"
    / "mitbih_test.csv"
)

MODEL_PATH = (
    PROJECT
    / "models"
    / "lstm_detector_heldout.pt"
)

THRESHOLD_PATH = (
    PROJECT
    / "models"
    / "lstm_threshold_heldout.npz"
)

FUSION_CONFIG_PATH = (
    PROJECT
    / "models"
    / "neurosymbolic_fusion.json"
)

OUT_PATH = (
    PROJECT
    / "results"
    / "model_comparison_heldout.csv"
)


# ============================================================
# PARAMETERS
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
# LSTM ARCHITECTURE
# EXACTLY MATCHES TRAINING CHECKPOINT
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
            1,
        )

        reconstructed, _ = self.decoder(
            decoder_input
        )

        return reconstructed, latent


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = LSTMAutoencoder(
        input_size=len(
            checkpoint["features"]
        ),
        hidden_size=checkpoint[
            "hidden_size"
        ],
        latent_size=checkpoint[
            "latent_size"
        ],
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# CREATE SEQUENCES
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

        group = group.reset_index(
            drop=True
        )

        values = (
            group[FEATURES]
            .astype(float)
            .values
        )

        values = (
            values - mean
        ) / np.where(
            std < 1e-8,
            1.0,
            std,
        )

        for end in range(
            SEQ_LEN - 1,
            len(group),
        ):

            start = (
                end - SEQ_LEN + 1
            )

            sequences.append(
                values[
                    start:end + 1
                ]
            )

            rows.append(
                group.iloc[end]
            )

    return (
        np.asarray(
            sequences,
            dtype=np.float32,
        ),
        pd.DataFrame(
            rows
        ).reset_index(
            drop=True
        ),
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_pred = np.asarray(
        y_pred
    ).astype(int)

    tp = int(
        (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (y_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()
    )

    accuracy = (
        (tp + tn)
        / max(
            tp + tn + fp + fn,
            1,
        )
    )

    precision = (
        tp
        / max(
            tp + fp,
            1,
        )
    )

    recall = (
        tp
        / max(
            tp + fn,
            1,
        )
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

    specificity = (
        tn
        / max(
            tn + fp,
            1,
        )
    )

    fpr = (
        fp
        / max(
            fp + tn,
            1,
        )
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "HELD-OUT MODEL COMPARISON"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required = [
        TEST_CSV,
        MODEL_PATH,
        THRESHOLD_PATH,
        FUSION_CONFIG_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print(
        "\nLoading held-out test dataset..."
    )

    df = pd.read_csv(
        TEST_CSV
    )

    test_df = df[
        df["record"].isin(
            TEST_RECORDS
        )
    ].copy()

    print(
        f"Test rows: {len(test_df)}"
    )

    for record in TEST_RECORDS:

        print(
            f"Record {record}: "
            f"{int((test_df['record'] == record).sum())} rows"
        )

    # --------------------------------------------------------
    # LOAD LSTM
    # --------------------------------------------------------

    print(
        "\nLoading frozen LSTM..."
    )

    model, checkpoint = load_model()

    mean = np.asarray(
        checkpoint["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["std"],
        dtype=np.float32,
    )

    print(
        f"Features: {len(checkpoint['features'])}"
    )

    print(
        f"Sequence length: "
        f"{checkpoint['sequence_length']}"
    )

    # --------------------------------------------------------
    # LOAD THRESHOLD
    # --------------------------------------------------------

    threshold_data = np.load(
        THRESHOLD_PATH
    )

    lstm_threshold = float(
        threshold_data["threshold"]
    )

    print(
        f"LSTM threshold: "
        f"{lstm_threshold:.6f}"
    )

    # --------------------------------------------------------
    # LOAD FUSION CONFIG
    # --------------------------------------------------------

    print(
        "\nLoading frozen fusion configuration..."
    )

    with open(
        FUSION_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        fusion_config = json.load(f)

    neural_weight = float(
        fusion_config[
            "neural_weight"
        ]
    )

    symbolic_weight = float(
        fusion_config[
            "symbolic_weight"
        ]
    )

    fusion_threshold = float(
        fusion_config[
            "decision_threshold"
        ]
    )

    print(
        f"Neural weight: "
        f"{neural_weight:.2f}"
    )

    print(
        f"Symbolic weight: "
        f"{symbolic_weight:.2f}"
    )

    print(
        f"Fusion threshold: "
        f"{fusion_threshold:.2f}"
    )

    # --------------------------------------------------------
    # CREATE COMMON SEQUENCES
    # --------------------------------------------------------

    print(
        "\nCreating common held-out sequences..."
    )

    X, rows = make_sequences(
        test_df,
        mean,
        std,
    )

    print(
        f"Total common sequences: "
        f"{len(X)}"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    y_true = (
        rows["annotation"]
        .astype(str)
        != "N"
    ).astype(int).values

    print(
        f"Normal sequences: "
        f"{int((y_true == 0).sum())}"
    )

    print(
        f"Non-N sequences: "
        f"{int((y_true == 1).sum())}"
    )

    # ========================================================
    # 1. LSTM DETECTOR
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "1. LSTM AUTOENCODER"
    )

    print(
        "=" * 70
    )

    X_tensor = torch.from_numpy(
        X
    )

    errors = []

    start_time = time.perf_counter()

    with torch.no_grad():

        batch_size = 256

        for start in range(
            0,
            len(X_tensor),
            batch_size,
        ):

            batch = X_tensor[
                start:start + batch_size
            ]

            reconstructed, _ = model(
                batch
            )

            mse = torch.mean(
                (
                    reconstructed
                    - batch
                ) ** 2,
                dim=(1, 2),
            )

            errors.extend(
                mse.numpy().tolist()
            )

    lstm_elapsed = (
        time.perf_counter()
        - start_time
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    lstm_prediction = (
        errors > lstm_threshold
    ).astype(int)

    lstm_metrics = metrics(
        y_true,
        lstm_prediction,
    )

    lstm_metrics[
        "model"
    ] = "LSTM"

    lstm_metrics[
        "sequences"
    ] = len(X)

    lstm_metrics[
        "latency_ms_total"
    ] = (
        lstm_elapsed * 1000
    )

    lstm_metrics[
        "latency_ms_per_sequence"
    ] = (
        lstm_elapsed
        * 1000
        / max(len(X), 1)
    )

    print(
        f"Accuracy:    "
        f"{lstm_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:   "
        f"{lstm_metrics['precision']:.4f}"
    )

    print(
        f"Recall:      "
        f"{lstm_metrics['recall']:.4f}"
    )

    print(
        f"F1:          "
        f"{lstm_metrics['f1']:.4f}"
    )

    print(
        f"Specificity: "
        f"{lstm_metrics['specificity']:.4f}"
    )

    print(
        f"FPR:         "
        f"{lstm_metrics['fpr']:.4f}"
    )

    # Neural risk for fusion
    neural_risk = np.minimum(
        errors / lstm_threshold,
        1.0,
    )

    # ========================================================
    # 2. SYMBOLIC DETECTOR
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "2. SYMBOLIC IDS"
    )

    print(
        "=" * 70
    )

    ids = NeuroSymbolicIDS()

    symbolic_prediction = []
    flag_counts = []

    symbolic_start = (
        time.perf_counter()
    )

    for _, row in rows.iterrows():

        features = {}

        for feature in FEATURES:

            value = row.get(
                feature
            )

            if pd.notna(value):

                features[
                    feature
                ] = float(value)

        detection = ids.evaluate(
            features,
            neural_score=0.0,
        )

        symbolic_prediction.append(
            1
            if detection.decision
            == "BLOCK"
            else 0
        )

        flag_counts.append(
            len(
                detection.flags
            )
        )

    symbolic_elapsed = (
        time.perf_counter()
        - symbolic_start
    )

    symbolic_prediction = np.asarray(
        symbolic_prediction,
        dtype=int,
    )

    flag_counts = np.asarray(
        flag_counts,
        dtype=float,
    )

    symbolic_metrics = metrics(
        y_true,
        symbolic_prediction,
    )

    symbolic_metrics[
        "model"
    ] = "Symbolic"

    symbolic_metrics[
        "sequences"
    ] = len(X)

    symbolic_metrics[
        "latency_ms_total"
    ] = (
        symbolic_elapsed * 1000
    )

    symbolic_metrics[
        "latency_ms_per_sequence"
    ] = (
        symbolic_elapsed
        * 1000
        / max(len(X), 1)
    )

    print(
        f"Accuracy:    "
        f"{symbolic_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:   "
        f"{symbolic_metrics['precision']:.4f}"
    )

    print(
        f"Recall:      "
        f"{symbolic_metrics['recall']:.4f}"
    )

    print(
        f"F1:          "
        f"{symbolic_metrics['f1']:.4f}"
    )

    print(
        f"Specificity: "
        f"{symbolic_metrics['specificity']:.4f}"
    )

    print(
        f"FPR:         "
        f"{symbolic_metrics['fpr']:.4f}"
    )

    # Symbolic risk
    symbolic_risk = np.minimum(
        flag_counts / 2.0,
        1.0,
    )

    # ========================================================
    # 3. NEURO-SYMBOLIC FUSION
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "3. NEURO-SYMBOLIC FUSION"
    )

    print(
        "=" * 70
    )

    fused_risk = (
        neural_weight
        * neural_risk
        + symbolic_weight
        * symbolic_risk
    )

    fusion_prediction = (
        fused_risk
        >= fusion_threshold
    ).astype(int)

    fusion_metrics = metrics(
        y_true,
        fusion_prediction,
    )

    fusion_metrics[
        "model"
    ] = "Neuro-Symbolic Fusion"

    fusion_metrics[
        "sequences"
    ] = len(X)

    # Fusion adds negligible arithmetic over the
    # already calculated neural and symbolic components.
    fusion_metrics[
        "latency_ms_total"
    ] = (
        lstm_elapsed
        + symbolic_elapsed
    ) * 1000

    fusion_metrics[
        "latency_ms_per_sequence"
    ] = (
        fusion_metrics[
            "latency_ms_total"
        ]
        / max(len(X), 1)
    )

    print(
        f"Accuracy:    "
        f"{fusion_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:   "
        f"{fusion_metrics['precision']:.4f}"
    )

    print(
        f"Recall:      "
        f"{fusion_metrics['recall']:.4f}"
    )

    print(
        f"F1:          "
        f"{fusion_metrics['f1']:.4f}"
    )

    print(
        f"Specificity: "
        f"{fusion_metrics['specificity']:.4f}"
    )

    print(
        f"FPR:         "
        f"{fusion_metrics['fpr']:.4f}"
    )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison_columns = [
        "model",
        "sequences",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
        "tp",
        "tn",
        "fp",
        "fn",
        "latency_ms_total",
        "latency_ms_per_sequence",
    ]

    comparison = pd.DataFrame(
        [
            lstm_metrics,
            symbolic_metrics,
            fusion_metrics,
        ]
    )

    comparison = comparison[
        comparison_columns
    ]

    # ========================================================
    # DISPLAY FINAL TABLE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL HELD-OUT COMPARISON"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUT_PATH,
        index=False,
    )

    print(
        "\nSaved comparison to:"
    )

    print(
        OUT_PATH
    )

    # ========================================================
    # IMPORTANT EXPERIMENT NOTE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT NOTE"
    )

    print(
        "=" * 70
    )

    print(
        "MIT-BIH N vs non-N annotations are used "
        "as a physiological anomaly proxy."
    )

    print(
        "They are NOT cyberattack ground truth."
    )

    print(
        "The test records were not used during "
        "fusion calibration."
    )

    print(
        "No test-time threshold tuning was performed."
    )

    print(
        "\nComparison experiment complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()