import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pqc_ring import LatticeRingPrototype
from src.chaos import coupled_sequence,shannon_entropy
from src.phase3 import aes_gcm_encrypt,aes_gcm_decrypt
from src.pipeline import BioTwin6GPipeline

def test_ring():
 a=LatticeRingPrototype(); r=[a.register(str(i)) for i in range(5)]; m=b'hello'; s=a.sign(m,r,2); assert a.verify(m,r,s)
def test_chaos(): assert len(coupled_sequence(b'x',32))==32; assert shannon_entropy(coupled_sequence(b'x',1024))>0
def test_aes():
 k=b'0'*32; n,c=aes_gcm_encrypt(k,b'abc'); assert aes_gcm_decrypt(k,n,c)==b'abc'
def test_pipeline():
 p=BioTwin6GPipeline(); r=p.process({'heart_rate':72,'glucose':115,'spo2':98,'packet_rate':10,'auth_failures':0,'jitter_ms':2,'neural_score':.1}); assert r.authenticated and r.ids_decision=='ALLOW'; a=p.process({'heart_rate':210,'glucose':340,'spo2':80,'packet_rate':2000,'auth_failures':5,'jitter_ms':50,'neural_score':.9}); assert a.ids_decision=='BLOCK'
