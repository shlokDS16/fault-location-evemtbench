"""Part 9 helper: per-window two-ended TD estimates on CIGRE MV with the IDENTIFIED pre-fault Z1.

Part 4 (mv_two_ended_zid.py) kept only the summary; this regenerates the per-window values with the same code
(mv.two_ended, identified R1 median per segment, nameplate X1) and adds a variant with identified R1 AND X1.
Output: results/validate_ha2_part9_mv_zid_perwindow.parquet (sim_idx, s1, td_np, td_idR, td_idRX).
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import mv  # noqa: E402

ZID = pd.read_csv(C.RESULTS / "validate_ha2_mv_zid.csv").set_index("line")


def work(r):
    if r["line"] not in ZID.index:
        return []
    sig = np.load(mv.CACHE / f"ep{r['sim_idx']}.npy")
    S, Rr = sig[:, 0:6], sig[:, 6:12]
    dS = savgol_filter(S[:, 0:3], 11, 2, deriv=1, delta=1.0 / C.FS, axis=0)
    dR = savgol_filter(Rr[:, 0:3], 11, 2, deriv=1, delta=1.0 / C.FS, axis=0)
    Z1, _ = mv.line_z(r["line"])
    R1 = float(ZID.loc[r["line"], "R1_id_med"])
    X1 = float(ZID.loc[r["line"], "X1_id_med"])
    nf = math.floor((r["start"] - 1.0) * C.FS)
    D = math.floor(r["dur"] * C.FS) if "incipient" in r["etype"] else 289
    out = []
    for s1 in range(C.WIN, len(sig) + 1, 48):
        if s1 > nf and s1 - C.WIN < nf + D:
            sl = slice(s1 - C.WIN, s1)
            out.append({"sim_idx": r["sim_idx"], "s1": s1,
                        "td_np": mv.two_ended(S, Rr, dS, dR, Z1.real, Z1.imag / mv.W0, sl),
                        "td_idR": mv.two_ended(S, Rr, dS, dR, R1, Z1.imag / mv.W0, sl),
                        "td_idRX": mv.two_ended(S, Rr, dS, dR, R1, X1 / mv.W0, sl)})
    return out


def main():
    e = mv.episodes()
    with ProcessPoolExecutor(max_workers=min(6, int(os.environ.get("VAL_WORKERS", "6")))) as ex:
        res = list(ex.map(work, e.to_dict("records"), chunksize=8))
    d = pd.DataFrame([x for rr in res for x in rr])
    d.to_parquet(C.RESULTS / "validate_ha2_part9_mv_zid_perwindow.parquet", index=False)
    print(len(d), d[["td_np", "td_idR", "td_idRX"]].isna().sum().to_dict(), flush=True)


if __name__ == "__main__":
    main()
