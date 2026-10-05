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
    "models/lstm_detector.pt"
)

DATASET_PATH = Path(
    "data/processed/mitbih_all.csv"
)

OUTPUT_PATH = Path(
    "models/lstm_threshold.npz"
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


def create_normal_sequences(
    df,
    features,
    sequence_length
):

    sequences = []

    # Keep record boundaries separate.
    for record in sorted(
        df["record"].astype(str).unique()
    ):

        record_df = df[
            df["record"].astype(str) == record
        ].copy()

        # Only reference-normal beats.
        record_df = record_df[
            record_df["annotation"] == "N"
        ].copy()

        if len(record_df) < sequence_length:
            continue

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

        for i in range(
            len(values) - sequence_length + 1
        ):

            sequences.append(
                values[
                    i:i + sequence_length
                ]
            )

    if not sequences:

        raise RuntimeError(
            "No normal sequences available."
        )

    return np.stack(
        sequences
    ).astype(np.float32)


def main():

    print("=" * 70)
    print(
        "BIO-TWIN 6G — LSTM THRESHOLD CALIBRATION"
    )
    print("=" * 70)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # -------------------------------------------------------------
    # Load model metadata
    # -------------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
	weights_only=False
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

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )

    normal_df = df[
        df["annotation"] == "N"
    ]

    # -------------------------------------------------------------
    # Normalize using training statistics
    # -------------------------------------------------------------

    normalized = df.copy()

    values = (
        normalized[features]
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

    normalized[features] = values

    # -------------------------------------------------------------
    # Create normal sequences
    # -------------------------------------------------------------

    X = create_normal_sequences(
        normalized,
        features,
        sequence_length
    )

    print()
    print(
        f"Normal beats       : {len(normal_df)}"
    )

    print(
        f"Normal sequences   : {len(X)}"
    )

    print(
        f"Sequence length    : {sequence_length}"
    )

    print(
        f"Features           : {len(features)}"
    )

    # -------------------------------------------------------------
    # Reconstruct
    # -------------------------------------------------------------

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

    tensor_x = torch.tensor(
        X,
        dtype=torch.float32
    )

    errors = []

    batch_size = 256

    with torch.no_grad():

        for start in range(
            0,
            len(tensor_x),
            batch_size
        ):

            batch = tensor_x[
                start:start + batch_size
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

    errors = np.asarray(
        errors,
        dtype=np.float64
    )

    # -------------------------------------------------------------
    # Threshold
    #
    # Use the 99th percentile of normal reconstruction error.
    # No abnormal labels are used.
    # -------------------------------------------------------------

    threshold = float(
        np.percentile(
            errors,
            99
        )
    )

    print()
    print("NORMAL RECONSTRUCTION ERROR")
    print("-" * 70)

    print(
        f"Mean             : "
        f"{errors.mean():.6f}"
    )

    print(
        f"Median           : "
        f"{np.median(errors):.6f}"
    )

    print(
        f"Std              : "
        f"{errors.std():.6f}"
    )

    print(
        f"95th percentile  : "
        f"{np.percentile(errors, 95):.6f}"
    )

    print(
        f"99th percentile  : "
        f"{threshold:.6f}"
    )

    print(
        f"Maximum          : "
        f"{errors.max():.6f}"
    )

    # -------------------------------------------------------------
    # Save threshold
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        OUTPUT_PATH,
        threshold=threshold,
        percentile=99,
        sequence_length=sequence_length,
        features=np.asarray(
            features
        )
    )

    print()
    print("-" * 70)

    print(
        f"Threshold saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 70)


if __name__ == "__main__":
    main()