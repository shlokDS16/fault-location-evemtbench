"""Part 6 helper: cache the REMOTE-terminal 6 channels (cubicle pex_MainBusY_<line>) of every DL / TG line-fault
episode and every adapt TEST line-fault episode -> results/validate_ha2_epcache/{DL,TG,AD}_R/ep<idx>.npy (4801 x 6).

Written from claudedocs/validator_spec_step3.md + c1_hybrid_design.md sections 21/22 only. Resumable.
The remote end is used ONLY by the two-ended locator, by the classical methods' remote source impedance (intended,
disclosed oracle) and by the measurement chain; it never reaches an H-A2 token.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NW = int(os.environ.get("VAL_WORKERS", "6"))


def remote_bus(line: str) -> str:
    m = re.fullmatch(r"MainLn(\d+)-(\d+)[AB]?", line)
    return f"MainBus{m.group(2)}"


def extract_remote(grid: str, sim_idx: int, line: str) -> str:
    out = C.CACHE / f"{grid}_R" / f"ep{sim_idx}.npy"
    if out.exists():
        return "cached"
    path = C.RAW / C.GRIDS[grid] / "data" / f"result{sim_idx}.csv"
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        names, units = next(rd), next(rd)
    tgt = f"pex_{remote_bus(line)}_{line}"
    cols = [j for j, nm in enumerate(names) if nm.endswith("\\" + tgt) or nm == tgt]
    if len(cols) != 6 or cols != list(range(cols[0], cols[0] + 6)):
        raise ValueError(f"{grid} {sim_idx}: remote cubicle {tgt}: {cols}")
    for k, ch in enumerate(C.CHANNELS):
        if ch not in units[cols[k]]:
            raise ValueError(f"{grid} {sim_idx}: unit {units[cols[k]]} != {ch}")
    raw = pd.read_csv(path, skiprows=2, header=None, usecols=[0] + cols, dtype=np.float64, engine="c").to_numpy()
    t = raw[:, 0]
    assert len(t) == C.N_SAMPLES and abs(t[0] - 1.0) < 1e-9, (grid, sim_idx)
    sig = np.ascontiguousarray(raw[:, 1:7])
    assert np.isfinite(sig).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, sig)
    tmp.replace(out)
    return "parsed"


def jobs() -> list[tuple[str, int, str]]:
    from ingrid import adapt_episodes  # noqa: E402
    j = []
    for g in ("DL", "TG"):
        for r in C.episodes(g).itertuples():
            j.append((g, int(r.sim_idx), r.line))
    a = adapt_episodes()
    for r in a[a["split"] == "test"].itertuples():
        j.append(("AD", int(r.sim_idx), r.line))
    return j


def main() -> None:
    j = jobs()
    print(len(j), "episodes", flush=True)
    with ProcessPoolExecutor(max_workers=NW) as ex:
        st = list(ex.map(extract_remote, *zip(*j), chunksize=8))
    print(pd.Series(st).value_counts().to_dict())


if __name__ == "__main__":
    main()
