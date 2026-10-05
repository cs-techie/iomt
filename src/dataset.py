from pathlib import Path
import pandas as pd
import numpy as np


def load_csv(path):
    return pd.read_csv(path)


def mitbih_available(folder):
    return sorted(str(p)[:-4] for p in Path(folder).glob("*.hea"))


def load_mitbih(record_path, beats=2000):
    """
    Load a MIT-BIH Arrhythmia Database record and extract
    beat-level physiological and ECG features.

    Output features:
        - record
        - beat_index
        - annotation
        - heart_rate
        - rr_interval
        - ecg_mean
        - ecg_std
        - ecg_min
        - ecg_max
    """

    try:
        import wfdb
    except ImportError:
        raise RuntimeError(
            "Install wfdb first: pip install wfdb"
        )

    record_path = Path(record_path)

    # Read ECG signal
    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    signal = rec.p_signal[:, 0]
    fs = float(rec.fs)

    # Keep annotations inside the available signal
    valid = [
        (sample, symbol)
        for sample, symbol in zip(ann.sample, ann.symbol)
        if sample < len(signal)
    ]

    rows = []

    # Need two consecutive beats to calculate RR interval
    max_beats = min(len(valid), beats + 1)

    for i in range(1, max_beats):

        current_sample, current_symbol = valid[i]
        previous_sample, _ = valid[i - 1]

        rr = (current_sample - previous_sample) / fs

                # Reject implausible RR intervals and annotation-edge artifacts.
        if rr < 0.3 or rr > 2.0:
            continue

        heart_rate = 60.0 / rr

        # Extract approximately one-second local ECG window
        start = max(0, current_sample - int(0.5 * fs))
        end = min(len(signal), current_sample + int(0.5 * fs))

        segment = signal[start:end]

        if len(segment) == 0:
            continue

        rows.append({
            "record": record_path.name,
            "beat_index": i,
            "annotation": current_symbol,
            "heart_rate": float(heart_rate),
            "rr_interval": float(rr),
            "ecg_mean": float(np.mean(segment)),
            "ecg_std": float(np.std(segment)),
            "ecg_min": float(np.min(segment)),
            "ecg_max": float(np.max(segment)),
        })

    return pd.DataFrame(rows)