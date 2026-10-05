import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.pipeline import BioTwin6GPipeline


DATASET = Path(
    "data/processed/mitbih_100.csv"
)


def main():

    print("=" * 70)
    print("BIO-TWIN 6G — MIT-BIH ECG PIPELINE")
    print("=" * 70)

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    print(f"Dataset rows : {len(df)}")
    print(f"Record       : {df['record'].iloc[0]}")
    print()

    pipeline = BioTwin6GPipeline()

    results = []

    # Process first 20 ECG beats
    for _, row in df.head(20).iterrows():

        result = pipeline.process_mitbih_row(
            row
        )

        results.append(result)

        print(
            f"Beat {int(row['beat_index']):4d} | "
            f"Annotation={row['annotation']:>2} | "
            f"HR={row['heart_rate']:7.2f} | "
            f"RR={row['rr_interval']:.4f} | "
            f"Decision={result.ids_decision:5s} | "
            f"Latency={result.latency_ms:7.3f} ms"
        )

    print()
    print("-" * 70)

    allowed = sum(
        r.ids_decision == "ALLOW"
        for r in results
    )

    blocked = sum(
        r.ids_decision == "BLOCK"
        for r in results
    )

    latencies = [
        r.latency_ms
        for r in results
    ]

    print(f"Processed beats : {len(results)}")
    print(f"ALLOW           : {allowed}")
    print(f"BLOCK           : {blocked}")
    print(
        f"Mean latency    : "
        f"{sum(latencies) / len(latencies):.3f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()