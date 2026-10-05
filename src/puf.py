import hashlib
import secrets
import random

class FuzzyExtractor:
    """
    Implements a Secure Sketch Fuzzy Extractor using a Repetition Code.
    Capable of correcting bit-flip noise from physical SRAM variations.
    """
    def __init__(self, key_bits=128, repetition=7):
        self.key_bits = key_bits
        self.repetition = repetition
        self.puf_bits = key_bits * repetition

    def _encode(self, key_bytes: bytes) -> list[int]:
        bits = []
        for byte in key_bytes:
            for i in range(8):
                bit = (byte >> i) & 1
                bits.extend([bit] * self.repetition)
        return bits

    def _decode(self, encoded_bits: list[int]) -> bytes:
        key_bytes = bytearray()
        for b in range(self.key_bits // 8):
            byte_val = 0
            for i in range(8):
                idx = (b * 8 + i) * self.repetition
                chunk = encoded_bits[idx : idx + self.repetition]
                # Majority vote error correction
                if sum(chunk) > self.repetition // 2:
                    byte_val |= (1 << i)
            key_bytes.append(byte_val)
        return bytes(key_bytes)

    def generate(self, r: list[int]) -> tuple[bytes, list[int]]:
        """Gen: Takes PUF response R, returns (Key, HelperData)"""
        k = secrets.token_bytes(self.key_bits // 8)
        c = self._encode(k)
        # Helper data W = R XOR C
        w = [r_bit ^ c_bit for r_bit, c_bit in zip(r, c)]
        return hashlib.sha256(k).digest(), w

    def reproduce(self, r_prime: list[int], w: list[int]) -> bytes:
        """Rep: Takes noisy PUF response R' and HelperData W, returns Key"""
        # C' = R' XOR W
        c_prime = [rp_bit ^ w_bit for rp_bit, w_bit in zip(r_prime, w)]
        k_prime = self._decode(c_prime)
        return hashlib.sha256(k_prime).digest()


class SimulatedSRAMPUF:
    """
    Simulates physical SRAM PUF cells with inherent noise (bit flips).
    Uses a Fuzzy Extractor to derive a stable cryptographic root key, 
    which is then used with HMAC to create a Challenge-Response interface.
    """
    def __init__(self, device_id: str, bit_error_rate=0.05):
        self.device_id = device_id
        self.extractor = FuzzyExtractor(key_bits=128, repetition=11)
        self.bit_error_rate = bit_error_rate
        
        # Simulate physical manufacturing variations (the true, stable PUF state)
        self._ground_truth_puf = [secrets.randbits(1) for _ in range(self.extractor.puf_bits)]
        self._helper_data = []  # Stored publicly on the device after enrollment

    def _read_sram(self) -> list[int]:
        """Simulates powering on the SRAM, introducing physical bit-flip noise."""
        return [
            bit ^ 1 if random.random() < self.bit_error_rate else bit 
            for bit in self._ground_truth_puf
        ]

    @classmethod
    def enroll(cls, device_id: str):
        """Run once during device provisioning in a secure facility."""
        obj = cls(device_id)
        r = obj._read_sram()
        # We discard the initial key, we only need to store the helper data
        _, helper_data = obj.extractor.generate(r)
        obj._helper_data = helper_data
        return obj

    def derive_key_material(self, challenge: bytes) -> bytes:
        """Run in the field: reconstructs the exact same key despite SRAM noise."""
        r_prime = self._read_sram()
        stable_root_key = self.extractor.reproduce(r_prime, self._helper_data)
        
        # Convert weak PUF to strong PUF Challenge-Response using HMAC
        import hmac
        return hmac.new(stable_root_key, self.device_id.encode() + b"|" + challenge, hashlib.sha256).digest()


