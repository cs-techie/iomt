import os
import sys
import math

import numpy as np
from scipy.special import gammaincc, erfc

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.chaos import coupled_sequence


def bytes_to_bits(data: bytes):
    return np.unpackbits(
        np.frombuffer(
            data,
            dtype=np.uint8
        )
    ).astype(np.int8)


def frequency_monobit(bits):
    n = len(bits)

    s = np.sum(
        2 * bits - 1
    )

    sobs = abs(s) / math.sqrt(n)

    p_value = erfc(
        sobs / math.sqrt(2)
    )

    return float(p_value)


def block_frequency(bits, block_size=128):
    n = len(bits)

    num_blocks = n // block_size

    if num_blocks == 0:
        return 0.0

    blocks = bits[
        :num_blocks * block_size
    ].reshape(
        num_blocks,
        block_size
    )

    proportions = np.mean(
        blocks,
        axis=1
    )

    chi_square = (
        4.0
        * block_size
        * np.sum(
            (proportions - 0.5) ** 2
        )
    )

    p_value = gammaincc(
        num_blocks / 2.0,
        chi_square / 2.0
    )

    return float(p_value)


def runs_test(bits):
    n = len(bits)

    pi = np.mean(bits)

    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0

    runs = 1 + np.sum(
        bits[1:] != bits[:-1]
    )

    numerator = abs(
        runs
        - 2.0 * n * pi * (1.0 - pi)
    )

    denominator = (
        2.0
        * math.sqrt(n)
        * math.sqrt(
            2.0 * pi * (1.0 - pi)
        )
    )

    if denominator == 0:
        return 0.0

    p_value = erfc(
        numerator / denominator
    )

    return float(p_value)


def longest_run_ones(bits):
    runs = []

    current = 0

    for bit in bits:

        if bit == 1:
            current += 1

        else:
            if current > 0:
                runs.append(current)

            current = 0

    if current > 0:
        runs.append(current)

    longest = max(runs) if runs else 0

    n = len(bits)

    expected = math.log2(n)

    deviation = abs(
        longest - expected
    )

    p_value = math.exp(
        -deviation / max(
            expected,
            1.0
        )
    )

    return float(
        min(1.0, p_value)
    )


def serial_test(bits, pattern_length=2):

    n = len(bits)

    if n <= pattern_length:
        return 0.0

    extended = np.concatenate(
        [
            bits,
            bits[:pattern_length - 1]
        ]
    )

    counts = {}

    for i in range(n):

        pattern = tuple(
            extended[
                i:i + pattern_length
            ]
        )

        counts[pattern] = (
            counts.get(pattern, 0)
            + 1
        )

    expected = (
        n /
        (2 ** pattern_length)
    )

    chi_square = sum(
        (
            count - expected
        ) ** 2 / expected
        for count in counts.values()
    )

    p_value = gammaincc(
        (2 ** pattern_length - 1) / 2.0,
        chi_square / 2.0
    )

    return float(p_value)


def approximate_entropy_test(
    bits,
    pattern_length=2
):

    n = len(bits)

    def phi(m):

        extended = np.concatenate(
            [
                bits,
                bits[:m - 1]
            ]
        )

        counts = {}

        for i in range(n):

            pattern = tuple(
                extended[
                    i:i + m
                ]
            )

            counts[pattern] = (
                counts.get(pattern, 0)
                + 1
            )

        total = 0.0

        for count in counts.values():

            p = count / n

            total += p * math.log(p)

        return total

    phi_m = phi(
        pattern_length
    )

    phi_m1 = phi(
        pattern_length + 1
    )

    approximate_entropy = (
        phi_m - phi_m1
    )

    chi_square = (
        2.0
        * n
        * (
            math.log(2)
            - approximate_entropy
        )
    )

    degrees = (
        2 ** pattern_length
    )

    p_value = gammaincc(
        degrees / 2.0,
        chi_square / 2.0
    )

    return float(p_value)


def run_test(name, function, bits):

    try:

        p_value = function(bits)

        passed = p_value >= 0.01

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{name:<28}"
            f"{p_value:>12.6f}"
            f"{status:>10}"
        )

        return passed

    except Exception as exc:

        print(
            f"{name:<28}"
            f"ERROR: {exc}"
        )

        return False


def main():

    sequence_length = 10000

    seed = (
        b"BioTwin6G-NIST-statistical-test"
    )

    print("=" * 60)
    print(
        "BIO-TWIN 6G RANDOMNESS TESTS"
    )
    print("=" * 60)

    print(
        f"Byte sequence length : "
        f"{sequence_length}"
    )

    data = coupled_sequence(
        seed,
        sequence_length
    )

    bits = bytes_to_bits(data)

    print(
        f"Bit sequence length  : "
        f"{len(bits)}"
    )

    print("-" * 60)

    print(
        f"{'Test':<28}"
        f"{'p-value':>12}"
        f"{'Status':>10}"
    )

    print("-" * 60)

    results = []

    results.append(
        run_test(
            "Frequency (Monobit)",
            frequency_monobit,
            bits
        )
    )

    results.append(
        run_test(
            "Block Frequency",
            block_frequency,
            bits
        )
    )

    results.append(
        run_test(
            "Runs",
            runs_test,
            bits
        )
    )

    results.append(
        run_test(
            "Longest Run of Ones",
            longest_run_ones,
            bits
        )
    )

    results.append(
        run_test(
            "Serial",
            serial_test,
            bits
        )
    )

    results.append(
        run_test(
            "Approximate Entropy",
            approximate_entropy_test,
            bits
        )
    )

    passed = sum(results)

    print("-" * 60)

    print(
        f"Tests passed : "
        f"{passed}/{len(results)}"
    )

    print(
        f"Pass rate    : "
        f"{100.0 * passed / len(results):.1f}%"
    )

    print(
        "Threshold    : p-value >= 0.01"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()