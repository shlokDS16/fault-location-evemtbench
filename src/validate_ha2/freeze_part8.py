"""Freeze my part-8 numbers (sha256 + timestamp) BEFORE opening any lead file named in validator_spec_part8.md."""
from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FILES = ["validate_ha2_part8_perwindow.parquet", "validate_ha2_part8_summary.csv", "validate_ha2_part8_nn_mv_runs.csv",
         "validate_ha2_part8_abl_summary.csv", "validate_ha2_part8_A3_ohm_DL.parquet", "validate_ha2_part8_A3_ohm_TG.parquet",
         "validate_ha2_part8_mv_windows.npz", "validate_ha2_part8.log", "validate_ha2_part8_nn_mv.log",
         "validate_ha2_part8_abl.log"]
FILES += sorted(p.name for p in C.RESULTS.glob("validate_ha2_part8_abl_pred_*.csv"))
FILES += sorted(p.name for p in C.RESULTS.glob("validate_ha2_nnpred_*_DLTGtoMV_s*.npy"))


def main() -> None:
    out = C.RESULTS / "validate_ha2_part8_frozen.sha256"
    assert not out.exists(), "already frozen"
    lines = []
    for f in FILES:
        p = C.RESULTS / f
        assert p.exists(), f
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  results/{f}")
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out.write_text(f"# frozen {ts} (before opening src/c1/review_checks.py, sync_subsample.py, adapt_td.py, "
                   f"results/c1_review_*, c1_adapt_td.csv, c1_ablations.csv, c1_zs_equal_info_MV.csv)\n"
                   + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
