"""Step 1: extract the 6 LOCAL-terminal channels of every line-fault episode to a per-episode cache.

Only the cubicle whose base column name ends in 'pex_MainBusX_<line>' (X = first bus number of the
faulted line) is kept. Channel order is verified against the units row (Isec A,B,C, Usec A,B,C).
Resumable: an episode whose .npy exists is skipped.
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def extract(grid: str, sim_idx: int, line: str) -> tuple[int, str]:
    out = C.CACHE / grid / f"ep{sim_idx}.npy"
    if out.exists():
        return sim_idx, "cached"
    path = C.RAW / C.GRIDS[grid] / "data" / f"result{sim_idx}.csv"
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        names = next(rd)
        units = next(rd)
    target = f"pex_{C.local_bus(line)}_{line}"
    cols = [j for j, nm in enumerate(names) if nm.endswith(target)]
    # guard against e.g. '..._MainLn1-2A' matching inside a longer name: require the separator
    cols = [j for j in cols if names[j].endswith("\\" + target) or names[j] == target]
    if len(cols) != 6 or cols != list(range(cols[0], cols[0] + 6)):
        raise ValueError(f"{grid} {sim_idx}: local cubicle columns for {target}: {cols}")
    for k, ch in enumerate(C.CHANNELS):
        if ch not in units[cols[k]]:
            raise ValueError(f"{grid} {sim_idx}: unit '{units[cols[k]]}' != {ch}")
    raw = pd.read_csv(path, skiprows=2, header=None, usecols=[0] + cols, dtype=np.float64,
                      engine="c").to_numpy()
    t = raw[:, 0]
    if len(t) != C.N_SAMPLES or abs(t[0] - 1.0) > 1e-9 or abs((t[-1] - t[0]) - 0.5) > 1e-6:
        raise ValueError(f"{grid} {sim_idx}: unexpected time axis len={len(t)} t0={t[0]} t1={t[-1]}")
    sig = np.ascontiguousarray(raw[:, 1:7])
    if not np.isfinite(sig).all():
        raise ValueError(f"{grid} {sim_idx}: non-finite samples")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, sig)
    tmp.replace(out)
    return sim_idx, "parsed"


def main() -> None:
    jobs = []
    for grid in ("DL", "TG"):
        ep = C.episodes(grid)
        for r in ep.itertuples():
            jobs.append((grid, int(r.sim_idx), r.line))
    print(f"{len(jobs)} episodes", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(extract, *j) for j in jobs]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 200 == 0:
                print(done, flush=True)
    print("done", done)


if __name__ == "__main__":
    main()
