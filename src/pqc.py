import pqcrypto.sign.ml_dsa_44 as mldsa
import pqcrypto.kem.ml_kem_768 as mlkem

class PQCProvider:
    """
    FIPS-204 (ML-DSA) and FIPS-203 (ML-KEM) Post-Quantum Cryptography.
    """
    def __init__(self, device_id: str):
        self.device_id = device_id
        # ML-DSA Signature Keypair
        self.pk, self.sk = mldsa.keygen()
        # ML-KEM Key Exchange Keypair
        self.kem_pk, self.kem_sk = mlkem.keygen()
        
    def sign(self, message: bytes) -> bytes:
        """Sign a message using ML-DSA (Dilithium)."""
        return mldsa.sign(self.sk, message)
        
    @staticmethod
    def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify an ML-DSA signature."""
        try:
            mldsa.verify(public_key, message, signature)
            return True
        except Exception:
            return False

    @staticmethod
    def encapsulate(target_pk: bytes) -> tuple[bytes, bytes]:
        """ML-KEM (Kyber) Key Encapsulation to generate a shared secret."""
        # Returns (ciphertext, shared_secret)
        return mlkem.encaps(target_pk)
        
    def decapsulate(self, ciphertext: bytes) -> bytes:
        """ML-KEM Decapsulation to recover the shared secret."""
        return mlkem.decaps(self.kem_sk, ciphertext)
