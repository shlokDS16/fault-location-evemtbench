"""Freeze part-7 results: sha256 of every results/validate_ha2_cond_* file + code + timestamp."""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

out = C.RESULTS / "validate_ha2_cond_frozen.sha256"
assert not out.exists(), "already frozen"
files = sorted(p for p in C.RESULTS.glob("validate_ha2_cond_*") if p.is_file() and p != out)
lines = [f"# frozen {datetime.now().astimezone().isoformat(timespec='seconds')} "
         "(before opening src/c1/conditioning.py, fig4_data.py, equal_info_preds.py, results/c1_cond*, "
         "results/c1_fig4*, results/c1_equal_info_preds*)"]
for p in files:
    lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
here = Path(__file__).resolve().parent
for n in ("cond.py", "freeze_cond.py"):
    lines.append(f"{hashlib.sha256((here / n).read_bytes()).hexdigest()}  src/validate_ha2/{n}")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
