"""Make the isolated subproject and shared read-only modules importable."""

from pathlib import Path
import sys

SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]
H1_ROOT = SUBPROJECT_ROOT.parent
WORKSPACE_ROOT = H1_ROOT

for path in (SUBPROJECT_ROOT, H1_ROOT / "src", WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
