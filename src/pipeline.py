import hashlib
import json
import time
import secrets

from dataclasses import dataclass

from .config import CONFIG
from .puf import SimulatedSRAMPUF
from .pqc import PQCProvider
from .privacy import add_gaussian_dp
from .digital_twin import BioDigitalTwin
from .phase2_ai import phase2_encode
from .phase3 import aes_gcm_encrypt, aes_gcm_decrypt
from .neuro_symbolic import NeuroSymbolicIDS
from .network import choose
from .blockchain import AuditChain
from .break_glass import BreakGlass
from .ssi import SSIRegistry


@dataclass
class Result:
    authenticated: bool
    ids_decision: str
    explanation: str
    delta: dict
    phase2_bytes: int
    final_bytes: int
    latency_ms: float
    slice_name: str
    slice_ok: bool
    entropy: float


class BioTwin6GPipeline:

    def __init__(self, ring_size=5):

        self.audit = AuditChain()

        self.puf = SimulatedSRAMPUF.enroll(
            "sensor-001"
        )

        self.pqc = PQCProvider("sensor-001")

        self.twin = BioDigitalTwin(
            "patient-001"
        )

        self.ids = NeuroSymbolicIDS()

        self.break_glass = BreakGlass(
            self.audit
        )

        self.ssi = SSIRegistry()

    def process(self, m, application="critical"):

        start = time.perf_counter()

        nonce = secrets.token_bytes(16)

        challenge = hashlib.sha256(
            json.dumps(
                m,
                sort_keys=True
            ).encode()
        ).digest()

        pufmat = self.puf.derive_key_material(
            challenge
        )

        vals = [
            m.get("heart_rate", 70),
            m.get("glucose", 100),
            m.get("spo2", 98)
        ]

        noisy, sigma = add_gaussian_dp(
            vals,
            CONFIG.dp_sensitivity,
            CONFIG.dp_epsilon,
            CONFIG.dp_delta
        )

        private = dict(m)

        private.update({
            "heart_rate": float(noisy[0]),
            "glucose": float(noisy[1]),
            "spo2": float(noisy[2])
        })

        payload = json.dumps(
            private,
            sort_keys=True
        ).encode()

        # Sign payload using ML-DSA (Dilithium)
        sig = self.pqc.sign(payload)

        authenticated = PQCProvider.verify(
            self.pqc.pk,
            payload,
            sig
        )

        det = self.ids.evaluate(
            private
        )

        self.audit.record({
            "type": "ids",
            "decision": det.decision,
            "flags": det.flags
        })

        if det.decision == "BLOCK":

            lat = (
                time.perf_counter() - start
            ) * 1000

            sl, ok = choose(
                application,
                lat
            )

            return Result(
                authenticated,
                det.decision,
                det.explanation,
                {},
                0,
                0,
                lat,
                sl.name,
                ok,
                0
            )

        delta = self.twin.update(
            private
        )

        p2 = phase2_encode(
            delta,
            pufmat,
            nonce
        )

        phase2 = p2["latent_cipher"]

        # ML-KEM (Kyber) Key Exchange for Phase 3 Payload encryption
        kem_ciphertext, shared_secret = PQCProvider.encapsulate(self.pqc.kem_pk)

        outer_key = hashlib.sha256(
            pufmat + shared_secret
        ).digest()

        n, c = aes_gcm_encrypt(
            outer_key,
            phase2,
            b"BioTwin6G|Phase3"
        )

        # Cloud-side ML-KEM decapsulation and decryption verification
        decapsulated_secret = self.pqc.decapsulate(kem_ciphertext)
        cloud_outer_key = hashlib.sha256(
            pufmat + decapsulated_secret
        ).digest()

        assert aes_gcm_decrypt(
            cloud_outer_key,
            n,
            c,
            b"BioTwin6G|Phase3"
        ) == phase2

        lat = (
            time.perf_counter() - start
        ) * 1000

        sl, ok = choose(
            application,
            lat
        )

        from .chaos import shannon_entropy

        ent = shannon_entropy(
            p2["chaos"]
        )

        self.audit.record({
            "type": "telemetry",
            "latency_ms": lat,
            "slice": sl.name,
            "delta_fields": list(delta),
            "final_bytes": len(c)
        })

        return Result(
            authenticated,
            det.decision,
            det.explanation,
            delta,
            len(phase2),
            len(c),
            lat,
            sl.name,
            ok,
            ent
        )

    def process_mitbih_row(
        self,
        row,
        application="critical"
    ):
        """
        Process one preprocessed MIT-BIH ECG beat.

        The ECG annotation is retained as experimental
        ground-truth metadata and is NOT inserted into
        the numeric Bio-Digital Twin state.
        """

        measurements = {
            "prev_rr": float(
                row["prev_rr"]
            ),

            "rolling_rr_mean": float(
                row["rolling_rr_mean"]
            ),

            "rolling_rr_std": float(
                row["rolling_rr_std"]
            ),

            "rr_deviation": float(
                row["rr_deviation"]
            ),

            "rr_deviation_z": float(
                row["rr_deviation_z"]
            ),

            "rolling_hr_mean": float(
                row["rolling_hr_mean"]
            ),

            "rolling_hr_std": float(
                row["rolling_hr_std"]
            ),

            "hr_deviation": float(
                row["hr_deviation"]
            ),

            "hr_deviation_z": float(
                row["hr_deviation_z"]
            ),
            "heart_rate": float(
                row["heart_rate"]
            ),

            "rr_interval": float(
                row["rr_interval"]
            ),

            "ecg_mean": float(
                row["ecg_mean"]
            ),

            "ecg_std": float(
                row["ecg_std"]
            ),

            "ecg_min": float(
                row["ecg_min"]
            ),

            "ecg_max": float(
                row["ecg_max"]
            )
        }

        result = self.process(
            measurements,
            application=application
        )

        return result