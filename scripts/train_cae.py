import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try: import torch, torch.nn as nn
except ImportError: raise SystemExit('Install optional PyTorch first.')
import numpy as np
from src.ai_models import CAE
rng=np.random.default_rng(42); X=torch.tensor(rng.normal(size=(2000,128)),dtype=torch.float32); m=CAE(128,16); opt=torch.optim.Adam(m.parameters(),1e-3); loss=nn.MSELoss()
for e in range(50): opt.zero_grad(); rec,z=m(X); l=loss(rec,X); l.backward(); opt.step()
out=Path('models/cae.pt'); out.parent.mkdir(exist_ok=True); torch.save(m.state_dict(),out); print('Saved',out,'reconstruction MSE',float(l))
