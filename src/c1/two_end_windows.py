"""Two-ended raw windows (design section 26a) for the official windows of one grid, built with EXACTLY the code path
that produced the paper's two-ended TD estimates (grid_pass.one: window_ends, Savitzky-Golay 11/2 derivative on the
whole episode, Clarke, nameplate R1/L1 from the network graph). Only windows with both terminals measured.
Per window: X12 (480 x 12: S then R terminal, Ia Ib Ic Va Vb Vc), UW (480 x 4: u_alpha, u_beta, w_alpha, w_beta of
Eq. (3)), Z (R1, X1, R0, X0 of the faulted line), y, sim_idx, tau (= post_ms of the grid parquet).
Usage: EVEMT_GRID=<DoubleLine|TestGrid110kV|CigreMVGrid> python src/c1/two_end_windows.py -> results/c1_raw2<TAG>.npz
Self-check: sum(u*w)/sum(w*w) over both modes from the saved UW must equal td2_est of results/c1_grid_<DL|TG|MV>.parquet."""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, TAG, terminal_cols  # noqa: E402
from grid_pass import LINE_RE, ROOT, W0, der, has_term, line_params, window_ends  # noqa: E402
from td_locator import clarke  # noqa: E402

GRIDTAG = {"DoubleLine": "DL", "TestGrid110kV": "TG", "CigreMVGrid": "MV"}


def one(args):
    row, LP = args
    line = row["events/event_target"]
    m = LINE_RE.match(line)
    sid = int(row["general/sim_idx"])
    f = D / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    flip = not has_term(hdr, m.group(1), line)
    if flip or not has_term(hdr, m.group(2), line):
        return None  # two-ended locator undefined (one terminal only)
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    XS = [d[c].to_numpy(float) for c in cs]
    XR = [d[c].to_numpy(float) for c in cr]
    nf, ends = window_ends(row, len(XS[0]))
    Z1, Z0, _ = LP[line]
    R, L = Z1.real, Z1.imag / W0
    iS, vS, iR, vR = clarke(*XS[:3]), clarke(*XS[3:]), clarke(*XR[:3]), clarke(*XR[3:])
    modes = [(vS[k] - vR[k] + R * iR[k] + L * der(iR[k]), R * (iS[k] + iR[k]) + L * (der(iS[k]) + der(iR[k])))
             for k in range(2)]
    UWfull = np.stack([modes[0][0], modes[1][0], modes[0][1], modes[1][1]], 1)
    X12 = np.stack(XS + XR, 1)
    y = row["events/event_flt_target_line_location"] / 100.0
    W, U, meta = [], [], []
    for s1 in ends:
        s0 = s1 - 480
        W.append(X12[s0:s1].astype(np.float32))
        U.append(UWfull[s0:s1].astype(np.float32))
        meta.append((sid, (s1 - nf) / FS * 1000, y, Z1.real, Z1.imag, Z0.real, Z0.imag))
    return np.stack(W), np.stack(U), meta


def main():
    gdir = D
    LP = line_params(gdir)
    s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(LINE_RE)
           & s["events/event_flt_target_line_location"].notna()]
    with ProcessPoolExecutor(4) as ex:
        res = [r for r in ex.map(one, [(r, LP) for _, r in fl.iterrows()], chunksize=4) if r is not None]
    n = sum(len(r[0]) for r in res)
    X12 = np.empty((n, 480, 12), np.float32)
    UW = np.empty((n, 480, 4), np.float32)
    M = np.array([m for r in res for m in r[2]], dtype=np.float64)
    i = 0
    for k, r in enumerate(res):
        X12[i:i + len(r[0])], UW[i:i + len(r[0])] = r[0], r[1]
        i += len(r[0])
        res[k] = None
    out = ROOT / "results" / f"c1_raw2{TAG}.npz"
    np.savez_compressed(out, X12=X12, UW=UW, Z=M[:, 3:7].astype(np.float32), y=M[:, 2], sim_idx=M[:, 0].astype(int), tau=M[:, 1])
    print(X12.shape, UW.shape, "->", out)

    # self-check against td2_est (grid parquet), aligned on sim_idx and window time; computed from the SAVED float32 UW
    g = pd.read_parquet(ROOT / "results" / f"c1_grid_{GRIDTAG[D.name]}.parquet", columns=["sim_idx", "post_ms", "td2_est"])
    g = g.dropna(subset=["td2_est"]).reset_index(drop=True)
    est = np.concatenate([(lambda A: (A[:, :, :2] * A[:, :, 2:]).sum((1, 2)) / ((A[:, :, 2:] ** 2).sum((1, 2)) + 1e-12))(
        UW[j:j + 2000].astype(np.float64)) for j in range(0, len(UW), 2000)])
    mine = pd.DataFrame(dict(sim_idx=M[:, 0].astype(int), post_ms=M[:, 1], est=est))
    j = g.merge(mine, on=["sim_idx", "post_ms"], how="outer", indicator=True)
    assert (j["_merge"] == "both").all() and len(j) == len(g) == len(mine), j["_merge"].value_counts()
    print(f"self-check {D.name}: n={len(j)}, max|est - td2_est| = {np.abs(j.est - j.td2_est).max():.3e}")


if __name__ == "__main__":
    main()
