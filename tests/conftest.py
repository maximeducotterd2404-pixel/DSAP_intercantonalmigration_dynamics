import sys
from pathlib import Path

# Add project root to PYTHONPATH so tests can import from src consistently.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
