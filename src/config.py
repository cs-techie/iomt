from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    lattice_degree: int = 256
    modulus_q: int = 2048
    ring_size: int = 5
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_sensitivity: float = 1.0
    key_bytes: int = 32
    aes_nonce_bytes: int = 12
    critical_iomt_budget_ms: float = 20.0
    tele_surgery_budget_ms: float = 10.0
    caE_input_dim: int = 128
    caE_latent_dim: int = 16
    seed: int = 42
CONFIG=Config()
