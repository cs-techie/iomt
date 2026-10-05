import os
import sys
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

from src.chaos import (
    logistic,
    tent,
    henon,
    chebyshev,
    lorenz,
    coupled_sequence,
    shannon_entropy
)


def bytes_from_values(values):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    values = np.mod(
        np.abs(values),
        1.0
    )

    return bytes(
        np.clip(
            values * 256.0,
            0,
            255
        ).astype(np.uint8)
    )


def lag1_correlation(data):
    if len(data) < 2:
        return 0.0

    x = np.frombuffer(
        data,
        dtype=np.uint8
    ).astype(np.float64)

    return float(
        np.corrcoef(
            x[:-1],
            x[1:]
        )[0, 1]
    )


def bit_balance(data):
    bits = np.unpackbits(
        np.frombuffer(
            data,
            dtype=np.uint8
        )
    )

    return float(
        np.mean(bits)
    )


def chi_square_test(data):

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


def generate_logistic(n):

    x = 0.371234

    output = []

    for _ in range(n + 1000):

        x = logistic(x)

        if len(output) < n and _ >= 1000:
            output.append(x)

    return bytes_from_values(output)


def generate_tent(n):

    x = 0.371234

    output = []

    for _ in range(n + 1000):

        x = tent(x)

        if len(output) < n and _ >= 1000:
            output.append(x)

    return bytes_from_values(output)


def generate_henon(n):

    x = 0.1
    y = 0.1

    output = []

    for i in range(n + 1000):

        x, y = henon(x, y)

        if i >= 1000:
            output.append(x)

    return bytes_from_values(output)


def generate_chebyshev(n):

    x = 0.371234

    output = []

    for _ in range(n + 1000):

        x = chebyshev(
            2.0 * x - 1.0
        )

        x = (x + 1.0) / 2.0

        if len(output) < n and _ >= 1000:
            output.append(x)

    return bytes_from_values(output)


def generate_lorenz(n):

    x = 0.1
    y = 0.1
    z = 0.1

    output = []

    for i in range(n + 1000):

        x, y, z = lorenz(
            x,
            y,
            z
        )

        if i >= 1000:
            output.append(
                abs(x) % 1.0
            )

    return bytes_from_values(output)


def evaluate(name, data):

    entropy = shannon_entropy(data)

    correlation = lag1_correlation(data)

    balance = bit_balance(data)

    chi, p_value = chi_square_test(data)

    print(
        f"{name:<12}"
        f"{entropy:>12.6f}"
        f"{correlation:>12.6f}"
        f"{balance:>12.6f}"
        f"{chi:>12.3f}"
        f"{p_value:>12.6f}"
    )


def main():

    n = 10000

    print("=" * 72)
    print("BIO-TWIN 6G FIVE-MAP CHAOS COMPARISON")
    print("=" * 72)

    print(
        f"{'Map':<12}"
        f"{'Entropy':>12}"
        f"{'Lag-1':>12}"
        f"{'Bit Mean':>12}"
        f"{'Chi2':>12}"
        f"{'p-value':>12}"
    )

    print("-" * 72)

    evaluate(
        "Logistic",
        generate_logistic(n)
    )

    evaluate(
        "Lorenz",
        generate_lorenz(n)
    )

    evaluate(
        "Hénon",
        generate_henon(n)
    )

    evaluate(
        "Tent",
        generate_tent(n)
    )

    evaluate(
        "Chebyshev",
        generate_chebyshev(n)
    )

    evaluate(
        "Coupled",
        coupled_sequence(
            b"BioTwin6G-five-map-comparison",
            n
        )
    )

    print("-" * 72)

    print(
        "Ideal entropy = 8 bits/byte"
    )

    print(
        "Ideal lag-1 correlation = 0"
    )

    print(
        "Ideal bit mean = 0.5"
    )

    print(
        "Chi-square p-value > 0.05 indicates"
    )

    print(
        "no significant deviation from uniformity"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()