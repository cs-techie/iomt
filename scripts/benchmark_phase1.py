import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from src.pqc_ring import LatticeRingPrototype
for size in [3,5,8,10,16]:
 a=LatticeRingPrototype(); ring=[a.register(str(i)) for i in range(size)]; vals=[a.benchmark(b'benchmark-message',ring,0) for _ in range(50)]; import numpy as np; print(size, 'sign',np.mean([v['sign_ms'] for v in vals]),'verify',np.mean([v['verify_ms'] for v in vals]),'total',np.mean([v['total_ms'] for v in vals]))
