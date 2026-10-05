import os
import sys
import time

import numpy as np
from scipy.stats import chi2

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.chaos import coupled_sequence, shannon_entropy


def lag1_correlation(data: bytes) -> float:
    if len(data) < 2:
        return 0.0

    x = np.frombuffer(
        data,
        dtype=np.uint8
    ).astype(np.float64)

    x1 = x[:-1]
    x2 = x[1:]

    if np.std(x1) == 0 or np.std(x2) == 0:
        return 0.0

    return float(
        np.corrcoef(x1, x2)[0, 1]
    )


def bit_balance(data: bytes) -> float:
    if not data:
        return 0.0

    bits = np.unpackbits(
        np.frombuffer(
            data,
            dtype=np.uint8
        )
    )

    return float(
        np.mean(bits)
    )


def chi_square_test(data: bytes):
    if not data:
        return 0.0, 1.0

    counts = np.bincount(
        np.frombuffer(
            data,
            dtype=np.uint8
        ),
        minlength=256
    )

    expected = len(data) / 256.0

    statistic = float(
        np.sum(
            (counts - expected) ** 2
            / expected
        )
    )

    p_value = float(
        chi2.sf(
            statistic,
            df=255
        )
    )

    return statistic, p_value


def benchmark(length: int, repeats: int = 5):

    seed = (
        b"BioTwin6G-chaos-statistical-benchmark"
    )

    # Warm-up
    coupled_sequence(
        seed,
        length
    )

    times = []

    for _ in range(repeats):

        start = time.perf_counter()

        data = coupled_sequence(
            seed,
            length
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        times.append(elapsed_ms)

    mean_ms = float(
        np.mean(times)
    )

    median_ms = float(
        np.median(times)
    )

    entropy = shannon_entropy(
        data
    )

    correlation = lag1_correlation(
        data
    )

    balance = bit_balance(
        data
    )

    chi, p_value = chi_square_test(
        data
    )

    return (
        mean_ms,
        median_ms,
        entropy,
        correlation,
        balance,
        chi,
        p_value
    )


def main():

    lengths = [
        32,
        256,
        1024,
        4096,
        10000
    ]

    print("=" * 100)
    print(
        "BIO-TWIN 6G CHAOS STATISTICAL BENCHMARK"
    )
    print("=" * 100)

    print(
        f"{'Length':>8} "
        f"{'Mean ms':>10} "
        f"{'Median ms':>10} "
        f"{'Entropy':>11} "
        f"{'Lag-1':>10} "
        f"{'Bit Mean':>10} "
        f"{'Chi2':>12} "
        f"{'p-value':>12}"
    )

    print("-" * 100)

    for length in lengths:

        (
            mean_ms,
            median_ms,
            entropy,
            correlation,
            balance,
            chi,
            chi_p
        ) = benchmark(length)

        print(
            f"{length:8d} "
            f"{mean_ms:10.3f} "
            f"{median_ms:10.3f} "
            f"{entropy:11.6f} "
            f"{correlation:10.6f} "
            f"{balance:10.6f} "
            f"{chi:12.3f} "
            f"{chi_p:12.6f}"
        )

    print("-" * 100)

    print(
        "Bit mean ideal ≈ 0.5"
    )

    print(
        "Lag-1 correlation ideal ≈ 0"
    )

    print(
        "Maximum byte-level Shannon entropy = 8 bits/byte"
    )

    print(
        "Chi-square p-value > 0.05: "
        "not statistically significant at the 5% level"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()