import json
import base64
import time
import uuid
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

class SSIRegistry:
    """
    Self-Sovereign Identity (SSI) Registry implementing W3C DIDs 
    and W3C Verifiable Credentials (VC) via JSON-LD formats and Ed25519 signatures.
    """
    
    def __init__(self):
        # In a real system, this would be a distributed ledger (e.g., Hyperledger Indy or IPFS)
        self.did_registry = {}
        
        # The Hospital/CA's master identity for issuing credentials
        self.issuer_private_key = ed25519.Ed25519PrivateKey.generate()
        self.issuer_public_key = self.issuer_private_key.public_key()
        self.issuer_did = self._generate_did(self.issuer_public_key)
        
        # Register the issuer
        self.did_registry[self.issuer_did] = self._create_did_document(self.issuer_did, self.issuer_public_key)

    def _generate_did(self, public_key):
        """Generates a did:key DID string based on the Ed25519 public key."""
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        # did:key format uses multicodec/multibase
        b64_key = base64.urlsafe_b64encode(pub_bytes).decode().rstrip('=')
        return f"did:key:{b64_key}"

    def _create_did_document(self, did, public_key):
        """Creates a W3C Standard DID Document"""
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        b64_key = base64.urlsafe_b64encode(pub_bytes).decode().rstrip('=')
        return {
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": did,
            "verificationMethod": [{
                "id": f"{did}#keys-1",
                "type": "Ed25519VerificationKey2020",
                "controller": did,
                "publicKeyMultibase": b64_key
            }],
            "authentication": [f"{did}#keys-1"]
        }

    def create_identity(self):
        """Creates a new decentralized identity and registers its DID document."""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        did = self._generate_did(public_key)
        
        self.did_registry[did] = self._create_did_document(did, public_key)
        return did, private_key

    def issue(self, role, license_id, issuer='Hospital-CA'):
        """Issues a Verifiable Credential conforming to the W3C VC Data Model."""
        # For the prototype pipeline compatibility, we automatically generate the subject's DID here
        subject_did, _ = self.create_identity()
        
        credential = {
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://example.org/examples/v1"
            ],
            "id": f"http://example.edu/credentials/{uuid.uuid4()}",
            "type": ["VerifiableCredential", "MedicalLicenseCredential"],
            "issuer": self.issuer_did,
            "issuanceDate": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "credentialSubject": {
                "id": subject_did,
                "role": role,
                "license_id_hash": hashlib.sha256(license_id.encode()).hexdigest()
            }
        }
        
        # Sign the credential
        payload = json.dumps(credential, sort_keys=True).encode()
        signature = self.issuer_private_key.sign(payload)
        
        # Attach proof (Ed25519Signature2020 style)
        credential["proof"] = {
            "type": "Ed25519Signature2020",
            "created": credential["issuanceDate"],
            "proofPurpose": "assertionMethod",
            "verificationMethod": f"{self.issuer_did}#keys-1",
            "jws": base64.urlsafe_b64encode(signature).decode()
        }
        
        # To maintain compatibility with existing pipeline code that expects a `c.did` and `c.role` attribute:
        class CompatibleCredential:
            def __init__(self, vc_dict):
                self.vc = vc_dict
                self.did = vc_dict["credentialSubject"]["id"]
                self.role = vc_dict["credentialSubject"]["role"]
        
        return CompatibleCredential(credential)

    def verify(self, c, role):
        """Verifies a Verifiable Credential cryptographically and logically."""
        if not hasattr(c, 'vc'):
            return False
            
        vc = c.vc
        
        # 1. Verify Logical Claims
        if vc["credentialSubject"]["role"] != role:
            return False
            
        issuer_did = vc["issuer"]
        if issuer_did not in self.did_registry:
            return False
            
        # 2. Extract Public Key of Issuer from DID Registry
        issuer_doc = self.did_registry[issuer_did]
        pub_b64 = issuer_doc["verificationMethod"][0]["publicKeyMultibase"]
        
        # Pad string to standard length if needed
        padding = '=' * (4 - (len(pub_b64) % 4))
        pub_bytes = base64.urlsafe_b64decode(pub_b64 + padding)
        
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        
        # 3. Verify Cryptographic Signature
        proof = vc.pop("proof")
        signature = base64.urlsafe_b64decode(proof["jws"] + padding)
        payload = json.dumps(vc, sort_keys=True).encode()
        
        # Restore proof to dict for integrity
        vc["proof"] = proof
        
        try:
            public_key.verify(signature, payload)
            return True
        except InvalidSignature:
            return False
