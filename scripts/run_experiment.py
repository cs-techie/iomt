import argparse,sys,json; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1])); import pandas as pd, numpy as np
from src.pipeline import BioTwin6GPipeline
p=argparse.ArgumentParser(); p.add_argument('--dataset',default=None); p.add_argument('--n',type=int,default=100); p.add_argument('--out',default='results/experiment_results.json'); a=p.parse_args(); pipe=BioTwin6GPipeline(); rows=[]
if a.dataset:
 df=pd.read_csv(a.dataset).head(a.n)
else:
 rng=np.random.default_rng(42); df=pd.DataFrame({'heart_rate':rng.normal(72,5,a.n),'glucose':rng.normal(115,12,a.n),'spo2':rng.normal(97,1,a.n),'packet_rate':rng.normal(12,2,a.n),'auth_failures':np.zeros(a.n),'jitter_ms':rng.normal(3,1,a.n),'neural_score':rng.normal(.08,.03,a.n)})
for _,x in df.iterrows():
 m={k:float(v) for k,v in x.items() if k in ['heart_rate','glucose','spo2','packet_rate','auth_failures','jitter_ms','neural_score']}; r=pipe.process(m); rows.append(r.__dict__)
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2)); lat=[x['latency_ms'] for x in rows]; print('Samples:',len(rows)); print('Mean latency (ms):',float(np.mean(lat))); print('P95 latency (ms):',float(np.percentile(lat,95))); print('Blocked:',sum(x['ids_decision']=='BLOCK' for x in rows)); print('Saved:',out)
