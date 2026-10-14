"""
Held-out Neuro-Symbolic IDS evaluation runner.
Delegates to evaluate_fusion.py for held-out evaluation on MIT-BIH records (107, 108, 109).
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.evaluate_fusion import main

if __name__ == "__main__":
    main()
