import pandas as pd
import numpy as np
import os

def prepare_wustl():
    print("=== WUSTL-EHMS 2020 Pipeline Preparer ===")
    print("This script prepares the WUSTL-EHMS dataset which uniquely combines biometric (ECG) data with network flow data.")
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "wustl_test.csv")
    
    # In a real environment, you would download the raw WUSTL-EHMS CSV and parse it here.
    # For this prototype, we simulate a mock 100-row dataframe conforming to WUSTL schemas.
    # WUSTL features: Dir, Flw, SrcIP, DstIP, SrcPort, DstPort, Proto, Flags, etc. + Biometrics
    
    data = []
    for i in range(100):
        is_attack = i > 80  # Last 20 rows are an attack
        row = {
            "record": 1,
            "timestamp": i * 0.5,
            # Network Flow Data (WUSTL)
            "packet_rate": np.random.normal(1200, 100) if is_attack else np.random.normal(10, 2),
            "auth_failures": 5 if is_attack else 0,
            "jitter_ms": np.random.normal(45, 10) if is_attack else np.random.normal(3, 1),
            # Biometric Data (WUSTL)
            "heart_rate": np.random.normal(190, 5) if is_attack else np.random.normal(70, 2),
            "blood_pressure": np.random.normal(160, 5) if is_attack else np.random.normal(120, 2),
            "spo2": np.random.normal(82, 1) if is_attack else np.random.normal(98, 1),
            "glucose": 100.0,
            # Truth label
            "annotation": "ATTACK" if is_attack else "N"
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)
    
    print(f"Successfully generated {len(df)} fused network-biometric records.")
    print(f"Saved to: {out_file}")
    print("You can now feed this into the Neuro-Symbolic IDS!")

if __name__ == "__main__":
    prepare_wustl()
