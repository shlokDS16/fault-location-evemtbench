"""Raw local-view windows (terminal S, 480 x 6: Ia Ib Ic Va Vb Vc) for the official windows of one grid, plus labels and
the faulted line's nameplate impedances (R1, X1, R0, X0 in ohm) for the equal-information baselines.
Usage: EVEMT_GRID=<grid> python src/c1/raw_windows.py -> results/c1_raw<TAG>.npz"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, T0, ROOT, TAG, terminal_cols  # noqa: E402
from local_features import WLEN, line_params  # noqa: E402


def one(args):
    row, LP = args
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    sid = int(row["general/sim_idx"])
    f = D / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs = terminal_cols(hdr, m.group(1), line)
    X = pd.read_csv(f, skiprows=[1], usecols=cs)[cs].to_numpy(np.float32)
    Z1, Z0 = LP[line]
    nf = int(round((row["events/event_start"] - T0) * FS))
    inc = "incipient" in row["events/event_type"]
    wins, meta = [], []
    for tau in (range(5, 51, 5) if inc else range(5, 81, 5)):
        s1 = nf + int(round(tau / 1000 * FS))
        wins.append(X[s1 - WLEN:s1])
        meta.append((sid, tau, row["events/event_flt_target_line_location"] / 100.0, Z1.real, Z1.imag, Z0.real, Z0.imag))
    return np.stack(wins), meta


def main():
    LP = line_params()
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, [(r, LP) for _, r in fl.iterrows()], chunksize=8))
    X = np.concatenate([r[0] for r in res])
    M = np.array([m for r in res for m in r[1]], dtype=np.float64)
    out = ROOT / "results" / f"c1_raw{TAG}.npz"
    np.savez(out, X=X, sim_idx=M[:, 0].astype(int), tau=M[:, 1], y=M[:, 2], Z=M[:, 3:7].astype(np.float32))
    print(X.shape, X.dtype, "->", out)


if __name__ == "__main__":
    main()
