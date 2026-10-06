"""Freeze part 9: sha256 of my output files and code, with a timestamp, BEFORE opening the lead's revamp_stats."""
import datetime
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FILES = [C.RESULTS / "validate_ha2_part9_summary.csv", C.RESULTS / "validate_ha2_part9_perwindow.parquet",
         C.RESULTS / "validate_ha2_part9_mv_zid_perwindow.parquet",
         Path(__file__).resolve().parent / "part9.py", Path(__file__).resolve().parent / "part9_mvzid.py"]
lines = [f"# frozen {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(C.ROOT).as_posix()}")
(C.RESULTS / "validate_ha2_part9_frozen.sha256").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
