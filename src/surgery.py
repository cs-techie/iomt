import numpy as np
def normal_commands(n=1000,dt=.01):
    t=np.arange(n)*dt; x=np.sin(.7*t); y=.5*np.cos(.5*t); z=.2*np.sin(.3*t); return np.column_stack([x,y,z])
def jitter_attack(cmd,jitter_ms=8,spike_every=80,seed=42):
    rng=np.random.default_rng(seed); out=cmd.copy(); out += rng.normal(0,jitter_ms/1000.0,out.shape)
    for i in range(spike_every,len(out),spike_every): out[i]+=rng.normal(0,.08,out.shape[1])
    return out

def smoothness_score(cmd):
    if len(cmd)<4:return 0
    vel=np.diff(cmd,axis=0); acc=np.diff(vel,axis=0); jerk=np.diff(acc,axis=0); return float(np.mean(np.linalg.norm(jerk,axis=1)))

from .pqc import PQCProvider
from .phase3 import aes_gcm_encrypt, aes_gcm_decrypt
import json

class SecureTeleSurgeryLink:
    """Simulates a secure ML-DSA signed and ML-KEM encrypted link for robot arm commands."""
    def __init__(self):
        self.pqc_doc = PQCProvider("doctor-arm")
        self.pqc_robot = PQCProvider("robot-arm")
        # Post-Quantum Key Exchange using Kyber/ML-KEM
        self.kem_ct, self.shared_secret = PQCProvider.encapsulate(self.pqc_robot.kem_pk)
        self.robot_secret = self.pqc_robot.decapsulate(self.kem_ct)
        assert self.shared_secret == self.robot_secret

    def transmit(self, cmd_row):
        # 1. Sign (Dilithium/ML-DSA)
        payload = json.dumps(cmd_row.tolist()).encode()
        sig = self.pqc_doc.sign(payload)
        
        # 2. Encrypt (AES-GCM with ML-KEM shared secret)
        packet = json.dumps({"payload": payload.decode(), "sig": sig.hex()}).encode()
        nonce, ciphertext = aes_gcm_encrypt(self.shared_secret, packet, b"tele-surgery")
        return nonce, ciphertext

    def receive(self, nonce, ciphertext):
        # 1. Decrypt
        packet_bytes = aes_gcm_decrypt(self.robot_secret, nonce, ciphertext, b"tele-surgery")
        if packet_bytes is None:
            raise ValueError("Decryption failed")
            
        packet = json.loads(packet_bytes.decode())
        payload = packet["payload"].encode()
        sig = bytes.fromhex(packet["sig"])
        
        # 2. Verify Signature
        if not PQCProvider.verify(self.pqc_doc.pk, payload, sig):
            raise ValueError("Signature invalid")
            
        return np.array(json.loads(payload.decode()))
