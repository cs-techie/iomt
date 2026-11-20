import sys
from pathlib import Path
import uvicorn

# Ensure repository root is on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    print("=" * 60)
    print("  Bio-Twin 6G: Architecture 2 Interactive Dashboard")
    print("  Serving at: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()

