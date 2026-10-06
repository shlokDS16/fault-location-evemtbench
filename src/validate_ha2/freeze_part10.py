"""Freeze my part-10 numbers (sha256 + timestamp) BEFORE opening any lead file named in validator_spec_part10.md."""
from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FILES = ["validate_ha2_part10_summary.csv", "validate_ha2_part10_A3_ohm_MV.parquet", "validate_ha2_part10.log",
         "validate_ha2_part10_pred_HA2_ref_MV.csv", "validate_ha2_part10_pred_A3_MV.csv",
         "validate_ha2_part10_pred_A1raw_MV.csv"]


def main() -> None:
    out = C.RESULTS / "validate_ha2_part10_frozen.sha256"
    assert not out.exists(), "already frozen"
    lines = []
    for f in FILES:
        p = C.RESULTS / f
        assert p.exists(), f
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  results/{f}")
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out.write_text(f"# frozen {ts} (before opening src/c1/ablation_a3_fix.py, ablations.py, zs_eval.py, "
                   f"results/c1_ablation_a3_fixed.csv, c1_ablations.csv)\n" + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
