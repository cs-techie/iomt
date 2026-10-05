import hashlib,hmac,secrets
class BreakGlass:
    """Proof-of-knowledge simulation; not a zk-SNARK."""
    def __init__(self,audit): self.audit=audit; self.revoked=set()
    def issue_doctor(self,role='doctor'):
        secret=secrets.token_bytes(32); commitment=hashlib.sha256(role.encode()+secret).hexdigest(); return secret,{'role':role,'commitment':commitment}
    def prove(self,secret,cred,challenge):
        expected=hashlib.sha256(cred['role'].encode()+secret).hexdigest()
        if not hmac.compare_digest(expected,cred['commitment']): raise ValueError('Credential mismatch')
        return hmac.new(secret,challenge,hashlib.sha256).hexdigest()
    def verify(self,secret,cred,challenge,proof,purpose):
        ok=cred['commitment'] not in self.revoked and hmac.compare_digest(hmac.new(secret,challenge,hashlib.sha256).hexdigest(),proof)
        self.audit.record({'type':'break_glass','purpose':purpose,'approved':ok,'commitment':cred['commitment']}); return ok
