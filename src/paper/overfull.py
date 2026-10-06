"""List overfull/underfull boxes of paper/main.log with the section file they come from.
Usage: python src/paper/overfull.py"""
import re
from pathlib import Path

log = (Path(__file__).resolve().parents[2] / "paper" / "main.log").read_text(encoding="latin-1")
cur = "main"
for line in log.splitlines():
    f = re.findall(r"\(\./sections/(\w+)\.tex", line)
    if f:
        cur = f[-1]
    o = re.search(r"(Overfull|Underfull) \\[hv]box \(([^)]*)\).*?lines? (\d+)", line)
    if o:
        print(f"{cur:14s} {o.group(1):9s} {o.group(2):28s} line {o.group(3)}")
