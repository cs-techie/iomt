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

from src.neuro_symbolic import NeuroSymbolicIDS


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_CSV = (
    PROJECT
    / "data"
    / "processed"
    / "mitbih_train.csv"
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

OUT_PATH = (
    PROJECT
    / "models"
    / "neurosymbolic_fusion.json"
)


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

SEQ_LEN = 20

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
# IMPORTANT:
# This architecture exactly matches the architecture used
# to create:
#
# models/lstm_detector_heldout.pt
#
# Do NOT add an output layer or change the decoder dimensions.
# ============================================================

class LSTMAutoencoder(torch.nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
        latent_size=16,
    ):
        super().__init__()

        # Encoder
        self.encoder = torch.nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
        )

        # Latent projection
        self.to_latent = torch.nn.Linear(
            hidden_size,
            latent_size,
        )

        # Latent expansion
        self.from_latent = torch.nn.Linear(
            latent_size,
            hidden_size,
        )

        # Decoder
        self.decoder = torch.nn.LSTM(
            hidden_size,
            input_size,
            batch_first=True,
        )

    def forward(self, x):

        # Encode sequence
        encoded, _ = self.encoder(x)

        # Use final encoder hidden representation
        last_hidden = encoded[:, -1, :]

        # Convert to latent representation
        latent = self.to_latent(
            last_hidden
        )

        # Expand latent representation
        decoder_input = self.from_latent(
            latent
        )

        # Add sequence dimension
        decoder_input = decoder_input.unsqueeze(1)

        # Repeat across the complete sequence
        decoder_input = decoder_input.repeat(
            1,
            x.size(1),
            1,
        )

        # Decode
        reconstructed, _ = self.decoder(
            decoder_input
        )

        return reconstructed, latent


# ============================================================
# LOAD LSTM MODEL
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

    # Create sequences separately for every record.
    #
    # This prevents a sequence from crossing from one
    # MIT-BIH record into another.
    #

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

        # Normalize using the statistics stored
        # in the held-out-training LSTM checkpoint.
        #
        # These statistics were calculated from
        # normal training data.

        values = (
            values - mean
        ) / np.where(
            std < 1e-8,
            1.0,
            std,
        )

        # Sequence length = 20
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

            # Keep the endpoint row.
            # Its annotation is used as the label
            # corresponding to this sequence.

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
# BINARY CLASSIFICATION METRICS
# ============================================================

def binary_metrics(
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
# MAIN CALIBRATION EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print(
        "NEURO-SYMBOLIC FUSION CALIBRATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    print(
        "\nLoading training dataset..."
    )

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_CSV}"
        )

    df = pd.read_csv(
        TRAIN_CSV
    )

    print(
        f"Training rows: {len(df)}"
    )

    print(
        "Training records:",
        ", ".join(
            str(x)
            for x in sorted(
                df["record"].unique()
            )
        ),
    )

    # --------------------------------------------------------
    # LOAD LSTM
    # --------------------------------------------------------

    print(
        "\nLoading held-out-training LSTM..."
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model not found:\n{MODEL_PATH}"
        )

    model, checkpoint = load_model()

    print(
        "Model features:",
        len(
            checkpoint["features"]
        ),
    )

    print(
        "Sequence length:",
        checkpoint[
            "sequence_length"
        ],
    )

    print(
        "Hidden size:",
        checkpoint[
            "hidden_size"
        ],
    )

    print(
        "Latent size:",
        checkpoint[
            "latent_size"
        ],
    )

    print(
        "Training annotation:",
        checkpoint[
            "training_annotation"
        ],
    )

    # --------------------------------------------------------
    # LOAD FROZEN LSTM THRESHOLD
    # --------------------------------------------------------

    print(
        "\nLoading frozen LSTM threshold..."
    )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"LSTM threshold not found:\n{THRESHOLD_PATH}"
        )

    threshold_data = np.load(
        THRESHOLD_PATH
    )

    threshold = float(
        threshold_data[
            "threshold"
        ]
    )

    print(
        "LSTM reconstruction threshold:",
        f"{threshold:.6f}",
    )

    # --------------------------------------------------------
    # NORMALIZATION PARAMETERS
    # --------------------------------------------------------

    mean = np.asarray(
        checkpoint["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["std"],
        dtype=np.float32,
    )

    # Safety check
    if len(mean) != len(FEATURES):
        raise ValueError(
            "Checkpoint mean length does not "
            "match the expected feature count."
        )

    if len(std) != len(FEATURES):
        raise ValueError(
            "Checkpoint std length does not "
            "match the expected feature count."
        )

    # --------------------------------------------------------
    # CREATE SEQUENCES
    # --------------------------------------------------------

    print(
        "\nCreating LSTM sequences..."
    )

    X, rows = make_sequences(
        df,
        mean,
        std,
    )

    print(
        f"Sequences available for fusion: "
        f"{len(X)}"
    )

    if len(X) == 0:
        raise RuntimeError(
            "No sequences were created."
        )

    # --------------------------------------------------------
    # LSTM RECONSTRUCTION ERRORS
    # --------------------------------------------------------

    print(
        "\nCalculating LSTM reconstruction errors..."
    )

    X_tensor = torch.from_numpy(
        X
    )

    errors = []

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

            # IMPORTANT:
            # The model returns:
            # reconstructed, latent

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

    errors = np.asarray(
        errors,
        dtype=float,
    )

    print(
        "Mean reconstruction error:",
        f"{errors.mean():.6f}",
    )

    print(
        "Median reconstruction error:",
        f"{np.median(errors):.6f}",
    )

    print(
        "95th percentile:",
        f"{np.percentile(errors, 95):.6f}",
    )

    print(
        "99th percentile:",
        f"{np.percentile(errors, 99):.6f}",
    )

    print(
        "Maximum reconstruction error:",
        f"{errors.max():.6f}",
    )

    # --------------------------------------------------------
    # NEURAL BINARY DECISION
    # --------------------------------------------------------

    neural_binary = (
        errors > threshold
    ).astype(int)

    print(
        "Neural anomaly predictions:",
        int(
            neural_binary.sum()
        ),
    )

    # --------------------------------------------------------
    # FROZEN SYMBOLIC DETECTOR
    # --------------------------------------------------------

    print(
        "\nRunning frozen symbolic rules..."
    )

    ids = NeuroSymbolicIDS()

    symbolic_binary = []
    flag_counts = []

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

        # IMPORTANT:
        #
        # neural_score is set to zero here.
        #
        # This ensures the symbolic detector is
        # evaluated independently of the LSTM.
        #
        detection = ids.evaluate(
            features,
            neural_score=0.0,
        )

        symbolic_binary.append(
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

    symbolic_binary = np.asarray(
        symbolic_binary,
        dtype=int,
    )

    flag_counts = np.asarray(
        flag_counts,
        dtype=float,
    )

    print(
        "Symbolic anomaly predictions:",
        int(
            symbolic_binary.sum()
        ),
    )

    print(
        "Rows with at least one symbolic flag:",
        int(
            (flag_counts > 0).sum()
        ),
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    #
    # IMPORTANT:
    # MIT-BIH annotation is used ONLY as a
    # normal-vs-non-N physiological anomaly proxy.
    #
    # It is NOT cyberattack ground truth.
    # --------------------------------------------------------

    y_true = (
        rows[
            "annotation"
        ]
        .astype(str)
        != "N"
    ).astype(int).values

    print(
        "\nTraining-label distribution:"
    )

    print(
        f"Normal (N): {(y_true == 0).sum()}"
    )

    print(
        f"Non-N:      {(y_true == 1).sum()}"
    )

    # --------------------------------------------------------
    # RISK NORMALIZATION
    # --------------------------------------------------------

    # Neural risk:
    #
    # reconstruction error equal to threshold
    # corresponds to risk = 1.
    #
    # Larger errors are clipped to 1.

    neural_risk = np.minimum(
        errors / threshold,
        1.0,
    )

    # Symbolic risk:
    #
    # 0 flags = 0 risk
    # 1 flag  = 0.5 risk
    # 2+ flags = 1.0 risk

    symbolic_risk = np.minimum(
        flag_counts / 2.0,
        1.0,
    )

    print(
        "\nNeural risk statistics:"
    )

    print(
        f"Mean:   {neural_risk.mean():.4f}"
    )

    print(
        f"Median: {np.median(neural_risk):.4f}"
    )

    print(
        "\nSymbolic risk statistics:"
    )

    print(
        f"Mean:   {symbolic_risk.mean():.4f}"
    )

    print(
        f"Median: {np.median(symbolic_risk):.4f}"
    )

    # --------------------------------------------------------
    # CALIBRATION SEARCH
    # --------------------------------------------------------

    candidates = []

    # Fixed transparent grid.
    #
    # Calibration happens ONLY on training records.
    #
    # Test records 107, 108 and 109 are NOT used here.

    weights = [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    print(
        "\nSearching weighted-risk configurations..."
    )

    for w in weights:

        fused_risk = (
            w * neural_risk
            + (1.0 - w)
            * symbolic_risk
        )

        for t in thresholds:

            prediction = (
                fused_risk >= t
            ).astype(int)

            metrics = binary_metrics(
                y_true,
                prediction,
            )

            candidates.append(
                {
                    "method": "weighted_risk",
                    "neural_weight": float(
                        w
                    ),
                    "symbolic_weight": float(
                        1.0 - w
                    ),
                    "decision_threshold": float(
                        t
                    ),
                    **metrics,
                }
            )

    # --------------------------------------------------------
    # OR BASELINE
    # --------------------------------------------------------

    print(
        "Evaluating OR baseline..."
    )

    or_prediction = (
        (neural_binary == 1)
        | (symbolic_binary == 1)
    ).astype(int)

    or_metrics = binary_metrics(
        y_true,
        or_prediction,
    )

    candidates.append(
        {
            "method": "OR",
            "neural_weight": None,
            "symbolic_weight": None,
            "decision_threshold": None,
            **or_metrics,
        }
    )

    # --------------------------------------------------------
    # AND BASELINE
    # --------------------------------------------------------

    print(
        "Evaluating AND baseline..."
    )

    and_prediction = (
        (neural_binary == 1)
        & (symbolic_binary == 1)
    ).astype(int)

    and_metrics = binary_metrics(
        y_true,
        and_prediction,
    )

    candidates.append(
        {
            "method": "AND",
            "neural_weight": None,
            "symbolic_weight": None,
            "decision_threshold": None,
            **and_metrics,
        }
    )

    # --------------------------------------------------------
    # SELECT BEST CONFIGURATION
    #
    # Primary criterion:
    # F1
    #
    # Tie-breakers:
    # recall
    # specificity
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x["f1"],
            x["recall"],
            x["specificity"],
        ),
        reverse=True,
    )

    best = candidates[0]

    # --------------------------------------------------------
    # DISPLAY BEST CONFIGURATION
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST TRAINING FUSION CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Method:              "
        f"{best['method']}"
    )

    if (
        best["method"]
        == "weighted_risk"
    ):

        print(
            f"Neural weight:       "
            f"{best['neural_weight']:.2f}"
        )

        print(
            f"Symbolic weight:     "
            f"{best['symbolic_weight']:.2f}"
        )

        print(
            f"Decision threshold:  "
            f"{best['decision_threshold']:.2f}"
        )

    print(
        f"Accuracy:             "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Precision:            "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1:                   "
        f"{best['f1']:.4f}"
    )

    print(
        f"Specificity:          "
        f"{best['specificity']:.4f}"
    )

    print(
        f"FPR:                  "
        f"{best['fpr']:.4f}"
    )

    print(
        f"TP:                   "
        f"{best['tp']}"
    )

    print(
        f"TN:                   "
        f"{best['tn']}"
    )

    print(
        f"FP:                   "
        f"{best['fp']}"
    )

    print(
        f"FN:                   "
        f"{best['fn']}"
    )

    # --------------------------------------------------------
    # SAVE FUSION CONFIGURATION
    # --------------------------------------------------------

    configuration = {

        "calibration_dataset":
            "mitbih_train.csv",

        "calibration_records":
            sorted(
                df[
                    "record"
                ]
                .unique()
                .tolist()
            ),

        "sequence_length":
            SEQ_LEN,

        "features":
            FEATURES,

        "lstm_model":
            "lstm_detector_heldout.pt",

        "lstm_threshold_file":
            "lstm_threshold_heldout.npz",

        "lstm_threshold":
            threshold,

        "method":
            best[
                "method"
            ],

        "neural_weight":
            best[
                "neural_weight"
            ],

        "symbolic_weight":
            best[
                "symbolic_weight"
            ],

        "decision_threshold":
            best[
                "decision_threshold"
            ],

        "training_metrics":
            best,

        "label_definition":
            (
                "N versus non-N MIT-BIH "
                "annotation; physiological "
                "anomaly proxy, not "
                "cyberattack ground truth."
            ),

        "neural_risk_definition":
            (
                "min(reconstruction_error / "
                "lstm_threshold, 1.0)"
            ),

        "symbolic_risk_definition":
            (
                "min(symbolic_flag_count / "
                "2.0, 1.0)"
            ),

        "calibration_grid":
            {
                "neural_weights":
                    weights,

                "decision_thresholds":
                    thresholds,
            },

        "test_records_reserved":
            [
                107,
                108,
                109,
            ],

        "test_records_used_during_calibration":
            False,
    }

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            configuration,
            f,
            indent=2,
        )

    print(
        "\nSaved calibration to:"
    )

    print(
        OUT_PATH
    )

    print(
        "\nCalibration complete."
    )

    print(
        "The configuration is now frozen "
        "for held-out test evaluation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()