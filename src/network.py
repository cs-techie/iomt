from dataclasses import dataclass
@dataclass(frozen=True)
class Slice: name:str; budget_ms:float; priority:int
SLICES={'tele_surgery':Slice('tele_surgery',10,100),'critical_iomt':Slice('critical_iomt',20,90),'normal_iomt':Slice('normal_iomt',100,50)}
def choose(app,latency):
    s=SLICES['tele_surgery' if app=='tele_surgery' else 'critical_iomt' if app=='critical' else 'normal_iomt']; return s,latency<=s.budget_ms
