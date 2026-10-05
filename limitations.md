# Project Limitations (Based on Current Codebase)

1. **WGAN-GP and CAE Models**: The WGAN-GP generator/critic classes and Convolutional Autoencoder (CAE) exist as code artifacts, but they are completely disconnected from the active pipeline. `train_cae.py` trains on pure random noise for just 50 steps, effectively inflating data rather than compressing it.
2. **Federated Learning Stub**: `train_federated.py` contains structural logic for a federated setup using `flwr`, but the local training loop is a commented-out stub. No actual federated gradient aggregation occurs.
3. **Tele-Surgery Mininet**: `scripts/mininet_6g_slice.py` exists to simulate slices using Mininet (Linux), but it has not been executed. Real packet loss, jitter under load, and bandwidth isolation are not currently simulated; latency is currently simulated via python's `perf_counter()`.
4. **Blockchain Environment**: The local AuditChain relies on a fallback "local simulation" state when Ganache is unavailable. It operates successfully in memory, but production EVM gas-fee measurements and block confirmation latencies are bypassed.
5. **Break-Glass Authentication**: The ZK-proof system relies on HMAC proof-of-knowledge rather than a full mathematical zero-knowledge implementation. The Solidity smart contract only flags a boolean license variable and does not implement advanced revocation or patient notification logic.
6. **Dataset Bias**: Evaluation is currently performed purely on MIT-BIH Arrhythmia records. MIT-BIH contains physiological anomalies, not cyber-attack patterns (which must be synthetically injected).
7. **Knowledge Graph Rules**: `neuro_symbolic.py` parses a minimal set of hardcoded threshold dictionaries rather than a complete OWL/RDF medical ontology.
