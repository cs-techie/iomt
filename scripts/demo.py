import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import BioTwin6GPipeline
from src.surgery import normal_commands, jitter_attack, smoothness_score, SecureTeleSurgeryLink
import secrets
p=BioTwin6GPipeline()
print('=== Bio-Twin 6G COMPLETE RESEARCH PROTOTYPE ===')
normal={'heart_rate':72,'glucose':118,'spo2':98,'packet_rate':12,'auth_failures':0,'jitter_ms':3,'neural_score':.08}
r=p.process(normal,'critical')
for k,v in r.__dict__.items(): print(f'{k:18}: {v}')
cred=p.ssi.issue('doctor','MED-LIC-001'); print('SSI DID           :',cred.did); print('SSI valid         :',p.ssi.verify(cred,'doctor'))
secret,c=p.break_glass.issue_doctor(); ch=secrets.token_bytes(32); proof=p.break_glass.prove(secret,c,ch); print('Break-glass        :',p.break_glass.verify(secret,c,ch,proof,'unconscious-patient')); print('Audit valid        :',p.audit.verify())
attack={'heart_rate':210,'glucose':340,'spo2':80,'packet_rate':2500,'auth_failures':8,'jitter_ms':55,'neural_score':.93}; a=p.process(attack,'tele_surgery'); print('\n=== ATTACK ==='); print('Decision           :',a.ids_decision); print('Explanation        :',a.explanation)
cmd=normal_commands(); atk=jitter_attack(cmd)
print('\n=== TELE-SURGERY SIMULATION ===')
link = SecureTeleSurgeryLink()
import time
start = time.perf_counter()
# Send 10 commands over secure link
for i in range(10):
    n, c = link.transmit(cmd[i])
    received = link.receive(n, c)
lat = (time.perf_counter() - start) * 1000 / 10
print(f'PQC Secure Link Latency: {lat:.2f} ms per command')
print('Normal jerk        :',smoothness_score(cmd))
print('Attacked jerk      :',smoothness_score(atk))
