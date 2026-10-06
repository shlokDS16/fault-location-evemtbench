"""Diagnostic: two-ended locator with the per-segment IDENTIFIED pre-fault R1 (median over episodes) instead of
nameplate R1 (X1 kept nameplate). Uses only pre-fault (normal operation) data for identification."""
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
    nf = math.floor((r["start"] - 1.0) * C.FS)
    D = math.floor(r["dur"] * C.FS) if "incipient" in r["etype"] else 289
    out = []
    for s1 in range(C.WIN, len(sig) + 1, 48):
        if s1 > nf and s1 - C.WIN < nf + D:
            d = mv.two_ended(S, Rr, dS, dR, R1, Z1.imag / mv.W0, slice(s1 - C.WIN, s1))
            out.append({"sim_idx": r["sim_idx"], "s1": s1, "two_ended_Rid": d})
    return out


def main():
    e = mv.episodes()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("VAL_WORKERS", "6"))) as ex:
        res = list(ex.map(work, e.to_dict("records"), chunksize=8))
    d = pd.DataFrame([x for rr in res for x in rr])
    p = pd.read_csv(C.RESULTS / "validate_ha2_mv_pred.csv").merge(d, on=["sim_idx", "s1"], validate="one_to_one")
    p["e_np"] = (p["two_ended"].clip(0, 1) - p["y"]).abs() * 100
    p["e_id"] = (p["two_ended_Rid"].clip(0, 1) - p["y"]).abs() * 100
    sc = p["ftype"].str.endswith("shc") | p["ftype"].str.endswith("shc_w_arc")
    out = {"n": len(p), "MAE_nameplate_R1": p["e_np"].mean(), "MAE_identified_R1": p["e_id"].mean(),
           "SC_MAE_nameplate_R1": p.loc[sc, "e_np"].mean(), "SC_MAE_identified_R1": p.loc[sc, "e_id"].mean()}
    for rf, g in p[sc & (p["pft_ms"] >= 25)].groupby("res"):
        out[f"SC_pft25_Rf{rf:g}_nameplate"] = g["e_np"].mean()
        out[f"SC_pft25_Rf{rf:g}_identified"] = g["e_id"].mean()
    s = pd.Series(out)
    s.to_csv(C.RESULTS / "validate_ha2_mv_two_ended_Rid.csv", header=["value"])
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
