from pathlib import Path
import sys

FULL64_ROOT = Path(__file__).resolve().parents[1]
REFINER_ROOT = FULL64_ROOT.parent
H1_ROOT = REFINER_ROOT.parent
WORKSPACE_ROOT = H1_ROOT

for path in (REFINER_ROOT, H1_ROOT / "src", WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

