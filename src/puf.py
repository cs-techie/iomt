import hashlib
import secrets
import random
# pyrefly: ignore [missing-import]
import bchlib

class BCHSecureSketch:
    """
    Implements a BCH-based Secure Sketch Fuzzy Extractor.
    Uses a concatenated code (Repetition + BCH) to tolerate 8-12% bit-flips
    from SRAM physical uncloneable functions.
    """
    def __init__(self, key_bytes=16, repetition=11, bch_t=12):
        self.key_bytes = key_bytes
        self.repetition = repetition
        
        # BCH protects the 128-bit key
        self.bch = bchlib.BCH(bch_t, m=8)
        
        # Dummy encode to find exact ECC length
        dummy_data = bytes([0] * self.key_bytes)
        self.ecc_bytes = len(self.bch.encode(dummy_data))
        self.c_bytes = self.key_bytes + self.ecc_bytes
        self.puf_bytes = self.c_bytes * self.repetition

    def _repeat(self, data: bytes) -> bytearray:
        out = bytearray()
        for b in data:
            for _ in range(self.repetition):
                out.append(b)
        return out

    def _unrepeat(self, data: bytearray) -> bytearray:
        out = bytearray()
        for i in range(0, len(data), self.repetition):
            chunk = data[i:i+self.repetition]
            # Bitwise majority vote across the repeated bytes
            reconstructed_byte = 0
            for bit_idx in range(8):
                bit_sum = 0
                for b in chunk:
                    bit_sum += (b >> bit_idx) & 1
                if bit_sum > self.repetition // 2:
                    reconstructed_byte |= (1 << bit_idx)
            out.append(reconstructed_byte)
        return out

    def generate(self, r: bytes) -> tuple[bytes, bytes]:
        if len(r) != self.puf_bytes:
            raise ValueError(f"PUF must be {self.puf_bytes} bytes")
            
        k = secrets.token_bytes(self.key_bytes)
        ecc = self.bch.encode(k)
        
        c = k + ecc
        # Repeat the ENTIRE codeword
        rep_c = self._repeat(c)
        
        # Helper data W = R XOR rep_C
        w = bytes([r_b ^ c_b for r_b, c_b in zip(r, rep_c)])
        return hashlib.sha256(k).digest(), w

    def reproduce(self, r_prime: bytes, w: bytes) -> bytes:
        # rep_C' = R' XOR W
        rep_c_prime = bytearray([rp_b ^ w_b for rp_b, w_b in zip(r_prime, w)])
        
        # First, unrepeat to correct massive bit-flips (reduces 10% -> ~1%)
        c_prime = self._unrepeat(rep_c_prime)
        
        data_corrupted = c_prime[:self.key_bytes]
        ecc_corrupted = c_prime[self.key_bytes:]
        
        bitflips = self.bch.decode(data_corrupted, ecc_corrupted)
        if bitflips == -1:
            raise ValueError("BCH decode failed: too many errors")
            
        self.bch.correct(data_corrupted, ecc_corrupted)
        return hashlib.sha256(data_corrupted).digest()


class SimulatedSRAMPUF:
    """
    Simulates physical SRAM PUF cells with inherent noise (bit flips).
    Uses a BCH Fuzzy Extractor to derive a stable cryptographic root key.
    """
    def __init__(self, device_id: str, bit_error_rate=0.10):
        self.device_id = device_id
        self.extractor = BCHSecureSketch(key_bytes=16, repetition=7, bch_t=12)
        self.bit_error_rate = bit_error_rate
        
        # Simulate physical manufacturing variations
        self._ground_truth_puf = secrets.token_bytes(self.extractor.puf_bytes)
        self._helper_data = b""

    def _read_sram(self) -> bytes:
        """Simulates powering on the SRAM, introducing physical bit-flip noise."""
        out = bytearray(self._ground_truth_puf)
        for i in range(len(out)):
            for bit in range(8):
                if random.random() < self.bit_error_rate:
                    out[i] ^= (1 << bit)
        return bytes(out)

    @classmethod
    def enroll(cls, device_id: str, bit_error_rate=0.10):
        """Run once during device provisioning in a secure facility."""
        obj = cls(device_id, bit_error_rate)
        # Enrollment uses the stable ground truth (simulating 
        # a clean laboratory read with averaging/burn-in testing)
        r = obj._ground_truth_puf
        _, helper_data = obj.extractor.generate(r)
        obj._helper_data = helper_data
        return obj

    def derive_key_material(self, challenge: bytes) -> bytes:
        """Run in the field: reconstructs the exact same key despite SRAM noise."""
        r_prime = self._read_sram()
        stable_root_key = self.extractor.reproduce(r_prime, self._helper_data)
        
        import hmac
        return hmac.new(stable_root_key, self.device_id.encode() + b"|" + challenge, hashlib.sha256).digest()


