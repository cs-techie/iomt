import hashlib
import numpy as np


# ============================================================
# Five chaotic maps
# ============================================================

def logistic(x, r=3.99):
    return r * x * (1.0 - x)


def tent(x, mu=1.999):
    return mu * x if x < 0.5 else mu * (1.0 - x)


def henon(x, y, a=1.4, b=0.3):
    return 1.0 - a * x * x + y, b * x


def chebyshev(x, k=5):
    x = np.clip(x, -1.0, 1.0)
    return np.cos(k * np.arccos(x))


def lorenz(x, y, z,
           sigma=10.0,
           rho=28.0,
           beta=8.0 / 3.0,
           dt=0.005):

    nx = x + sigma * (y - x) * dt
    ny = y + (x * (rho - z) - y) * dt
    nz = z + (x * y - beta * z) * dt

    return nx, ny, nz


# ============================================================
# Deterministic initial state
# ============================================================

def _initial_state(seed: bytes):

    digest = hashlib.sha512(seed).digest()

    v = np.frombuffer(
        digest,
        dtype=np.uint8
    ).astype(np.float64) / 255.0

    x = 0.01 + 0.98 * v[0]
    y = 0.01 + 0.98 * v[1]
    z = 0.01 + 0.98 * v[2]

    hx = 0.2 * v[3] - 0.1
    hy = 0.2 * v[4] - 0.1

    return x, y, z, hx, hy


# ============================================================
# Lightweight coupled multi-chaos generator
# ============================================================

def coupled_sequence(seed: bytes, n=256):

    if n <= 0:
        return b""

    x, y, z, hx, hy = _initial_state(seed)

    output = bytearray()

    # Small burn-in: enough to move away from initial state
    # without creating excessive computation.
    burn_in = 128

    # Generate a modest internal pool and then expand it
    # deterministically using SHA-256.
    pool_size = max(64, min(128, n))

    for i in range(burn_in + pool_size):

        # Logistic
        x = logistic(
            (x + 0.13 * (abs(hx) % 1.0)) % 1.0
        )
        x %= 1.0

        # Lorenz
        x, y, z = lorenz(x, y, z)

        x %= 1.0
        y %= 1.0
        z %= 1.0

        # Henon
        hx, hy = henon(hx, hy)

        # Keep Henon numerically bounded
        hx = np.tanh(hx)
        hy = np.tanh(hy)

        # Tent
        tt = tent(
            (x + 0.17 * (abs(hy) % 1.0)) % 1.0
        )
        tt %= 1.0

        # Chebyshev
        cc_input = (
            2.0 * ((y + 0.11 * tt) % 1.0) - 1.0
        )

        cc = chebyshev(cc_input)

        # Coupling
        x = (
            x
            + 0.07 * (abs(z) % 1.0)
            + 0.05 * tt
        ) % 1.0

        y = (
            y
            + 0.06 * (abs(hx) % 1.0)
            + 0.04 * (abs(cc) % 1.0)
        ) % 1.0

        z = (
            z
            + 0.04 * tt
            + 0.03 * (abs(hy) % 1.0)
        ) % 1.0

        if i < burn_in:
            continue

        # Mix all five chaotic components.
        value = (
            0.23 * x
            + 0.17 * y
            + 0.19 * z
            + 0.13 * (abs(hx) % 1.0)
            + 0.11 * (abs(hy) % 1.0)
            + 0.09 * tt
            + 0.08 * (abs(cc) % 1.0)
        )

        value = (value * 997.0) % 1.0

        output.append(
            int(value * 256.0) & 0xFF
        )

    raw = bytes(output)

    # Lightweight deterministic whitening.
    #
    # This is statistical post-processing, NOT a claim that
    # the chaos generator itself is cryptographically secure.
    result = bytearray()

    counter = 0

    while len(result) < n:

        block = hashlib.sha256(
            seed
            + raw
            + counter.to_bytes(4, "big")
        ).digest()

        result.extend(block)

        counter += 1

    return bytes(result[:n])


# ============================================================
# Shannon entropy
# ============================================================

def shannon_entropy(data: bytes):

    if not data:
        return 0.0

    values = np.frombuffer(
        data,
        dtype=np.uint8
    )

    counts = np.bincount(
        values,
        minlength=256
    )

    probabilities = counts[counts > 0] / len(values)

    return float(
        -(probabilities * np.log2(probabilities)).sum()
    )