import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit(
        "PyTorch is not installed. Install it with: pip install torch"
    )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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

SEQUENCE_LENGTH = 20

DEFAULT_OUT = "models/lstm_detector.pt"


# ---------------------------------------------------------------------
# LSTM Autoencoder
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------

def create_sequences(
    df,
    features,
    sequence_length
):

    sequences = []

    # IMPORTANT:
    # Create sequences separately for every record.
    # Never allow a sequence to cross a record boundary.

    for record in sorted(
        df["record"].astype(str).unique()
    ):

        record_df = df[
            df["record"].astype(str) == record
        ].copy()

        # Train only on reference-normal N beats.
        record_df = record_df[
            record_df["annotation"] == "N"
        ].copy()

        if len(record_df) <= sequence_length:
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
            "No training sequences were created."
        )

    return np.stack(
        sequences
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a record-aware LSTM "
            "autoencoder on MIT-BIH reference-normal beats."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Input processed MIT-BIH CSV"
    )

    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output PyTorch model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    csv_path = Path(args.csv)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    required = [
        "record",
        "annotation",
    ] + FEATURES

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # -------------------------------------------------------------
    # Select features
    # -------------------------------------------------------------

    available_features = [
        feature
        for feature in FEATURES
        if feature in df.columns
    ]

    print("=" * 70)
    print(
        "BIO-TWIN 6G — LSTM AUTOENCODER TRAINING"
    )
    print("=" * 70)

    print()
    print(
        f"Dataset        : {csv_path}"
    )

    print(
        f"Records        : "
        f"{df['record'].nunique()}"
    )

    print(
        f"Total rows     : "
        f"{len(df)}"
    )

    print(
        f"Features       : "
        f"{len(available_features)}"
    )

    print(
        f"Sequence length: "
        f"{args.sequence_length}"
    )

    # -------------------------------------------------------------
    # Compute normalization from N beats only
    # -------------------------------------------------------------

    normal_df = df[
        df["annotation"] == "N"
    ].copy()

    normal_values = (
        normal_df[available_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )

    mean = normal_values.mean(
        axis=0
    )

    std = (
        normal_values.std(
            axis=0
        ) + 1e-6
    )

    # Apply normalization before sequence creation.
    normalized = df.copy()

    normalized[
        available_features
    ] = (
        normalized[
            available_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
        - mean
    ) / std

    # -------------------------------------------------------------
    # Create record-safe sequences
    # -------------------------------------------------------------

    X = create_sequences(
        normalized,
        available_features,
        args.sequence_length
    )

    print()
    print(
        f"Training sequences: {len(X)}"
    )

    # -------------------------------------------------------------
    # Torch dataset
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device             : {device}"
    )

    tensor_x = torch.tensor(
        X,
        dtype=torch.float32
    )

    dataset = torch.utils.data.TensorDataset(
        tensor_x
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = LSTMAutoencoder(
        input_size=len(available_features),
        hidden_size=32,
        latent_size=16
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.MSELoss()

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    print()
    print("TRAINING")
    print("-" * 70)

    model.train()

    for epoch in range(
        1,
        args.epochs + 1
    ):

        epoch_loss = 0.0
        samples = 0

        for (batch,) in loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed, _ = model(
                batch
            )

            loss = criterion(
                reconstructed,
                batch
            )

            loss.backward()

            optimizer.step()

            batch_size = batch.size(0)

            epoch_loss += (
                loss.item() * batch_size
            )

            samples += batch_size

        epoch_loss /= max(
            samples,
            1
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):

            print(
                f"Epoch {epoch:02d}/"
                f"{args.epochs}  "
                f"Loss = {epoch_loss:.6f}"
            )

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    output = Path(
        args.out
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": available_features,
            "mean": mean,
            "std": std,
            "sequence_length":
                args.sequence_length,
            "hidden_size": 32,
            "latent_size": 16,
            "training_annotation": "N",
            "records": sorted(
                df["record"]
                .astype(str)
                .unique()
            ),
        },
        output
    )

    print()
    print("-" * 70)
    print(
        f"Model saved to: {output}"
    )

    print(
        f"Final training loss: "
        f"{epoch_loss:.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()