"""Freeze part-6 results: sha256 of every results/validate_ha2_step3_* file + code + timestamp."""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

out = C.RESULTS / "validate_ha2_step3_frozen.sha256"
assert not out.exists(), "already frozen"
files = sorted(p for p in C.RESULTS.glob("validate_ha2_step3_*") if p.is_file() and p != out)
lines = [f"# frozen {datetime.now().astimezone().isoformat(timespec='seconds')} "
         "(before opening src/c1/, results/c1_classical* or results/c1_chain*)"]
for p in files:
    lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
here = Path(__file__).resolve().parent
for n in ("remote.py", "classical.py", "chain.py", "freeze_step3.py"):
    lines.append(f"{hashlib.sha256((here / n).read_bytes()).hexdigest()}  src/validate_ha2/{n}")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
