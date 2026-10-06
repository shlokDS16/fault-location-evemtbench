"""Freeze part-5 results: sha256 of every results/validate_ha2_guard_* file + timestamp."""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

out = C.RESULTS / "validate_ha2_guard_frozen.sha256"
assert not out.exists(), "already frozen"
files = sorted(p for p in C.RESULTS.glob("validate_ha2_guard_*") if p.is_file() and p != out)
lines = [f"# frozen {datetime.now().astimezone().isoformat(timespec='seconds')} "
         "(before opening src/c1/, results/c1_* or results/rerun_guard*)"]
for p in files:
    lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
for p in (Path(__file__).resolve().parent / "guard.py", Path(__file__).resolve()):
    lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  src/validate_ha2/{p.name}")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
