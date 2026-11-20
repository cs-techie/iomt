import sys
from pathlib import Path
import json
import time
import secrets
import hashlib
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root folder is on path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.pipeline import BioTwin6GPipeline
from src.surgery import normal_commands, jitter_attack, smoothness_score, SecureTeleSurgeryLink
from src.pqc import PQCProvider
from src.network import SLICES, choose
from src.privacy import add_gaussian_dp
from src.config import CONFIG
from src.phase2_ai import phase2_encode
from src.phase3 import aes_gcm_encrypt, aes_gcm_decrypt
from src.chaos import shannon_entropy

app = FastAPI(
    title="Bio-Twin 6G Dashboard",
    description="Interactive UI & API for Bio-Twin 6G Architecture 2 Prototype",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent singleton pipeline
PIPELINE = BioTwin6GPipeline()
TELE_LINK = None

def get_tele_link():
    global TELE_LINK
    if TELE_LINK is None:
        TELE_LINK = SecureTeleSurgeryLink()
    return TELE_LINK

class PipelineRunRequest(BaseModel):
    heart_rate: float = 72.0
    glucose: float = 118.0
    spo2: float = 98.0
    blood_pressure: float = 120.0
    packet_rate: float = 12.0
    auth_failures: int = 0
    jitter_ms: float = 3.0
    neural_score: float = 0.08
    slice_name: str = "critical"  # "tele_surgery", "critical", or "normal"

class SurgerySimRequest(BaseModel):
    num_commands: int = 50
    jitter_ms: float = 8.0
    spike_every: int = 20

class SSIIssueRequest(BaseModel):
    role: str = "doctor"
    license_id: str = "MED-LIC-001"

class BreakGlassRequest(BaseModel):
    role: str = "doctor"
    purpose: str = "unconscious-patient-cardiac-arrest"

@app.get("/api/status")
def get_system_status():
    return {
        "status": "online",
        "device_id": PIPELINE.puf.device_id,
        "puf_enrolled": True,
        "pqc_algorithms": {
            "signature": "ML-DSA-44 (Dilithium / FIPS-204)",
            "kem": "ML-KEM-768 (Kyber / FIPS-203)"
        },
        "digital_twin_id": PIPELINE.twin.patient_id,
        "blockchain_connected": PIPELINE.audit.connected,
        "audit_blocks_count": len(PIPELINE.audit.blocks),
        "audit_chain_valid": PIPELINE.audit.verify(),
        "available_slices": {k: {"budget_ms": v.budget_ms, "priority": v.priority} for k, v in SLICES.items()}
    }

@app.post("/api/pipeline/run")
def run_pipeline(req: PipelineRunRequest):
    start = time.perf_counter()
    vitals = req.model_dump()
    slice_type = vitals.pop("slice_name", "critical")
    
    # 1. Hardware PUF Key Material Derivation
    challenge = hashlib.sha256(json.dumps(vitals, sort_keys=True).encode()).digest()
    pufmat = PIPELINE.puf.derive_key_material(challenge)
    
    # 2. Gaussian DP
    raw_vals = [vitals.get("heart_rate", 70.0), vitals.get("glucose", 100.0), vitals.get("spo2", 98.0)]
    noisy_vals, sigma = add_gaussian_dp(raw_vals, CONFIG.dp_sensitivity, CONFIG.dp_epsilon, CONFIG.dp_delta)
    
    private_vitals = dict(vitals)
    private_vitals.update({
        "heart_rate": float(noisy_vals[0]),
        "glucose": float(noisy_vals[1]),
        "spo2": float(noisy_vals[2])
    })
    
    # 3. PQC Authentication (ML-DSA-44)
    payload_bytes = json.dumps(private_vitals, sort_keys=True).encode()
    sig = PIPELINE.pqc.sign(payload_bytes)
    pqc_valid = PQCProvider.verify(PIPELINE.pqc.pk, payload_bytes, sig)
    
    # 4. Neuro-Symbolic IDS
    det = PIPELINE.ids.evaluate(private_vitals)
    
    PIPELINE.audit.record({
        "type": "ids",
        "decision": det.decision,
        "flags": det.flags,
        "neural_score": vitals.get("neural_score", 0.0)
    })
    
    # If blocked, stop here
    if det.decision == "BLOCK":
        total_lat = (time.perf_counter() - start) * 1000
        sl, sl_ok = choose(slice_type, total_lat)
        return {
            "authenticated": pqc_valid,
            "ids_decision": det.decision,
            "explanation": det.explanation,
            "flags": det.flags,
            "delta": {},
            "puf_challenge_hex": challenge.hex()[:16] + "...",
            "puf_key_preview": pufmat.hex()[:16] + "...",
            "dp": {
                "raw": {"heart_rate": raw_vals[0], "glucose": raw_vals[1], "spo2": raw_vals[2]},
                "noisy": {"heart_rate": round(private_vitals["heart_rate"], 2), "glucose": round(private_vitals["glucose"], 2), "spo2": round(private_vitals["spo2"], 2)},
                "sigma": sigma
            },
            "pqc": {
                "algorithm": "ML-DSA-44",
                "valid": pqc_valid,
                "sig_bytes": len(sig),
                "sig_preview": sig.hex()[:32] + "..."
            },
            "phase2": None,
            "phase3": None,
            "latency_ms": round(total_lat, 2),
            "slice": {
                "name": sl.name,
                "budget_ms": sl.budget_ms,
                "sla_met": sl_ok
            },
            "entropy": 0.0
        }
        
    # 5. Twin Update & Delta Sync
    delta = PIPELINE.twin.update(private_vitals)
    
    # 6. Phase 2 Chaos + CAE
    nonce = secrets.token_bytes(16)
    p2 = phase2_encode(delta, pufmat, nonce)
    phase2_cipher = p2["latent_cipher"]
    ent = shannon_entropy(p2["chaos"])
    
    # 7. Phase 3 ML-KEM + AES-256-GCM
    kem_ciphertext, shared_secret = PQCProvider.encapsulate(PIPELINE.pqc.kem_pk)
    outer_key = hashlib.sha256(pufmat + shared_secret).digest()
    n_iv, ciphertext = aes_gcm_encrypt(outer_key, phase2_cipher, b"BioTwin6G|Phase3")
    
    # Verify cloud decapsulation
    decaps_secret = PIPELINE.pqc.decapsulate(kem_ciphertext)
    cloud_key = hashlib.sha256(pufmat + decaps_secret).digest()
    decrypted_check = aes_gcm_decrypt(cloud_key, n_iv, ciphertext, b"BioTwin6G|Phase3")
    assert decrypted_check == phase2_cipher
    
    total_lat = (time.perf_counter() - start) * 1000
    sl, sl_ok = choose(slice_type, total_lat)
    
    PIPELINE.audit.record({
        "type": "telemetry",
        "slice": sl.name,
        "latency_ms": total_lat,
        "phase2_bytes": len(phase2_cipher),
        "final_bytes": len(ciphertext)
    })
    
    return {
        "authenticated": pqc_valid,
        "ids_decision": det.decision,
        "explanation": det.explanation,
        "flags": det.flags,
        "delta": {k: round(v, 2) if isinstance(v, float) else v for k, v in delta.items()},
        "puf_challenge_hex": challenge.hex()[:16] + "...",
        "puf_key_preview": pufmat.hex()[:16] + "...",
        "dp": {
            "raw": {"heart_rate": raw_vals[0], "glucose": raw_vals[1], "spo2": raw_vals[2]},
            "noisy": {"heart_rate": round(private_vitals["heart_rate"], 2), "glucose": round(private_vitals["glucose"], 2), "spo2": round(private_vitals["spo2"], 2)},
            "sigma": sigma
        },
        "pqc": {
            "algorithm": "ML-DSA-44",
            "valid": pqc_valid,
            "sig_bytes": len(sig),
            "sig_preview": sig.hex()[:32] + "..."
        },
        "phase2": {
            "bytes": len(phase2_cipher),
            "entropy": round(ent, 3),
            "chaos_map": "5-map composite chaos (Logistic, Sine, Chebyshev, Circle, Gauss)"
        },
        "phase3": {
            "algorithm": "AES-256-GCM + ML-KEM-768",
            "final_bytes": len(ciphertext),
            "iv_preview": n_iv.hex(),
            "cipher_preview": ciphertext.hex()[:32] + "...",
            "kem_verified": True
        },
        "latency_ms": round(total_lat, 2),
        "slice": {
            "name": sl.name,
            "budget_ms": sl.budget_ms,
            "sla_met": sl_ok
        },
        "entropy": round(ent, 3)
    }

@app.post("/api/surgery/simulate")
def simulate_surgery(req: SurgerySimRequest):
    link = get_tele_link()
    n = min(max(req.num_commands, 10), 100)
    
    cmd = normal_commands(n=n)
    atk = jitter_attack(cmd, jitter_ms=req.jitter_ms, spike_every=req.spike_every)
    
    # Measure transmission latency over secure PQC link
    start = time.perf_counter()
    transmitted_samples = 10
    for i in range(min(transmitted_samples, n)):
        nonce, ct = link.transmit(cmd[i])
        rx = link.receive(nonce, ct)
    pqc_lat = ((time.perf_counter() - start) * 1000) / transmitted_samples
    
    normal_jerk = smoothness_score(cmd)
    attacked_jerk = smoothness_score(atk)
    
    # Return trajectory slices for 2D/3D plotting (X, Y, Z coordinates)
    pts_normal = [{"x": round(float(row[0]), 4), "y": round(float(row[1]), 4), "z": round(float(row[2]), 4)} for row in cmd]
    pts_attacked = [{"x": round(float(row[0]), 4), "y": round(float(row[1]), 4), "z": round(float(row[2]), 4)} for row in atk]
    
    return {
        "num_commands": n,
        "pqc_latency_ms_per_cmd": round(pqc_lat, 2),
        "normal_jerk": float(f"{normal_jerk:.6e}"),
        "attacked_jerk": round(attacked_jerk, 5),
        "smoothness_degradation": round(attacked_jerk / (normal_jerk + 1e-9), 1),
        "trajectory_normal": pts_normal,
        "trajectory_attacked": pts_attacked
    }

@app.post("/api/ssi/issue")
def ssi_issue(req: SSIIssueRequest):
    cred = PIPELINE.ssi.issue(req.role, req.license_id)
    is_valid = PIPELINE.ssi.verify(cred, req.role)
    
    return {
        "did": cred.did,
        "role": cred.role,
        "license": cred.license,
        "verified": is_valid,
        "issuer_did": PIPELINE.ssi.issuer_did,
        "raw_credential": {
            "@context": cred.vc.get("@context"),
            "id": cred.vc.get("id"),
            "type": cred.vc.get("type"),
            "issuer": cred.vc.get("issuer"),
            "issuanceDate": cred.vc.get("issuanceDate"),
            "credentialSubject": cred.vc.get("credentialSubject"),
            "proof": {
                "type": cred.vc.get("proof", {}).get("type"),
                "verificationMethod": cred.vc.get("proof", {}).get("verificationMethod"),
                "jws": cred.vc.get("proof", {}).get("jws", "")[:32] + "..."
            }
        }
    }

@app.post("/api/break-glass/trigger")
def break_glass_trigger(req: BreakGlassRequest):
    secret, cred = PIPELINE.break_glass.issue_doctor(req.role)
    challenge = secrets.token_bytes(32)
    proof = PIPELINE.break_glass.prove(secret, cred, challenge)
    ok = PIPELINE.break_glass.verify(secret, cred, challenge, proof, req.purpose)
    
    return {
        "role": req.role,
        "purpose": req.purpose,
        "commitment": cred["commitment"],
        "challenge_hex": challenge.hex()[:16] + "...",
        "proof_preview": proof[:24] + "...",
        "verified": ok,
        "audit_valid": PIPELINE.audit.verify(),
        "total_audit_blocks": len(PIPELINE.audit.blocks)
    }

@app.get("/api/audit/logs")
def get_audit_logs():
    blocks_out = []
    for b in PIPELINE.audit.blocks[-15:]:
        blocks_out.append({
            "index": b.index,
            "timestamp": time.strftime('%H:%M:%S', time.localtime(b.timestamp)),
            "event": b.event,
            "hash": b.hash[:16] + "...",
            "prev_hash": b.previous_hash[:16] + "..."
        })
    return {
        "total_blocks": len(PIPELINE.audit.blocks),
        "chain_intact": PIPELINE.audit.verify(),
        "recent_blocks": list(reversed(blocks_out))
    }

@app.get("/api/mitbih/sample")
def get_mitbih_sample():
    csv_path = ROOT_DIR / "data" / "processed" / "mitbih_100.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = pd.read_csv(csv_path)
    sample_df = df.head(30)
    records = sample_df.to_dict(orient="records")
    return {
        "total_records": len(df),
        "records": records
    }

# Read and serve UI template
INDEX_HTML_PATH = Path(__file__).resolve().parent / "templates" / "index.html"

@app.get("/", response_class=HTMLResponse)
def serve_index():
    if not INDEX_HTML_PATH.exists():
        return HTMLResponse("<h1>UI Template not found. Please ensure templates/index.html exists.</h1>", status_code=500)
    return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))

