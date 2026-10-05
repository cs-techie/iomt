import json, numpy as np
from .chaos import coupled_sequence

def bytes_to_vector(b,n=128):
    x=np.frombuffer((b*((n+len(b)-1)//len(b)))[:n],dtype=np.uint8).astype(np.float32)/255.0
    return x

def latent_xor(data:bytes,key:bytes):
    kb=(key*((len(data)+len(key)-1)//len(key)))[:len(data)]
    return bytes(a^b for a,b in zip(data,kb))

def phase2_encode(delta, puf_material, session_nonce):
    raw=json.dumps(delta,sort_keys=True,separators=(',',':')).encode()
    chaos=coupled_sequence(puf_material+session_nonce,256)
    session_key=coupled_sequence(chaos+puf_material+session_nonce,32)
    # Lightweight deterministic compression surrogate for the runnable baseline.
    # Optional CAE training script can replace this representation.
    padded=raw
    cipher=latent_xor(padded,session_key)
    return {'raw':raw,'chaos':chaos,'session_key':session_key,'latent_cipher':cipher}
