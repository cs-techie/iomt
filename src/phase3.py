from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

def aes_gcm_encrypt(key,plaintext,aad=b'BioTwin6G'):
    nonce=secrets.token_bytes(12); return nonce,AESGCM(key).encrypt(nonce,plaintext,aad)
def aes_gcm_decrypt(key,nonce,ciphertext,aad=b'BioTwin6G'): return AESGCM(key).decrypt(nonce,ciphertext,aad)
