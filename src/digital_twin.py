from dataclasses import dataclass,field
import numpy as np
@dataclass
class BioDigitalTwin:
    patient_id:str
    state:dict[str,float]=field(default_factory=dict)
    version:int=0
    def update(self,m):
        delta={}
        for k,v in m.items():
            v=float(v); old=self.state.get(k)
            if old is None: delta[k]=v
            else: delta[k]=v-old
            self.state[k]=v
        self.version+=1; return delta
    def apply_delta(self,d):
        for k,v in d.items(): self.state[k]=self.state.get(k,0)+float(v)
        self.version+=1
    def sync_error(self,target):
        keys=set(target)|set(self.state); return float(np.mean([abs(self.state.get(k,0)-target.get(k,0)) for k in keys]))
