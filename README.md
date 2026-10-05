# Bio-Twin 6G — Complete Research Prototype

This project implements the faculty-proposed **Architecture 2** as a laptop-runnable research prototype.
It preserves the original paper's three cryptographic phases and adds the proposed Bio-Twin/6G extensions.

## Architecture implemented

**Four functional layers**
1. Hardware & Network Trust: simulated SRAM-PUF, KDF, lattice-ring authentication abstraction, 6G slice simulation.
2. Bio-Digital Twin & Secure Data: preprocessing, delta synchronization, Gaussian DP, Phase 1→2→3 secure pipeline.
3. SSI & Access Control: DID/credential simulation and break-glass proof-of-knowledge + audit-chain simulation.
4. Neuro-Symbolic Threat Detection: LSTM anomaly score (optional trained model) + medical/operational rules + explanations.

**Three cryptographic phases**
- Phase 1: lattice-based ring-signature abstraction (NTT-inspired benchmark; NOT production PQC).
- Phase 2: five-map chaos + optional WGAN-GP session-key model + CAE latent compression/encryption.
- Phase 3: AES-256-GCM authenticated encryption.

## Important research honesty

`src/pqc_ring.py` is deliberately a **research abstraction**, not a claim of production post-quantum security. A real paper should replace it with a vetted PQC implementation and report the exact standardized primitive/parameter set. The same applies to the PUF and ZK components: this repository simulates their behavior so the whole system can be evaluated on a normal laptop.

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv venv
.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\demo.py
python -m pytest tests -q
```

## Dataset workflow

The faculty document proposes MIT-BIH Arrhythmia as a primary ECG source. Put the PhysioNet MIT-BIH record files under `data/raw/mitdb/` (for example `100.dat`, `100.hea`, `100.atr`). Do not rename the record extensions.

Then:

```powershell
python scripts\prepare_mitbih.py --record 100 --beats 2000
python scriptsun_experiment.py --dataset data/processed/mitbih_100.csv
```

The IDS can also be evaluated on a network-traffic CSV. Expected feature names can include `heart_rate, glucose, spo2, packet_rate, auth_failures, jitter_ms, neural_score, label`.

## Optional AI training

The complete pipeline runs without PyTorch. If PyTorch is installed, the scripts can train a small LSTM anomaly detector and CAE/WGAN-GP components.

```powershell
python scripts	rain_cae.py
python scripts	rain_lstm.py --csv data/processed/mitbih_100.csv
```

The current code intentionally separates **research prototype components** from claims of real cryptographic security.
