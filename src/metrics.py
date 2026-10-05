import time, numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def classification_report(y_true,y_pred):
    return {'accuracy':accuracy_score(y_true,y_pred),'precision':precision_score(y_true,y_pred,zero_division=0),'recall':recall_score(y_true,y_pred,zero_division=0),'f1':f1_score(y_true,y_pred,zero_division=0),'confusion_matrix':confusion_matrix(y_true,y_pred).tolist()}
def timed(fn,*args,**kwargs):
    t=time.perf_counter(); out=fn(*args,**kwargs); return out,(time.perf_counter()-t)*1000

def npcr(a,b):
    a=np.frombuffer(a,dtype=np.uint8); b=np.frombuffer(b,dtype=np.uint8); n=min(len(a),len(b)); return float(np.mean(a[:n]!=b[:n])*100)
def uaci(a,b):
    a=np.frombuffer(a,dtype=np.uint8).astype(float); b=np.frombuffer(b,dtype=np.uint8).astype(float); n=min(len(a),len(b)); return float(np.mean(np.abs(a[:n]-b[:n])/255)*100)
