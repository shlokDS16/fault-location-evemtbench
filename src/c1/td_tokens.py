"""C1 design section 15: time-domain one-ended tokens (local terminal S only) for every official window.
Per loop: d_td (LS on [u, Delta i]), d_rl (LS on u only, i.e. R_f = 0) and the relative residual of the 2-column fit.
Samples used: [s0 + N, s1) so that Delta x(t) = x(t) - x(t - N) is defined inside the window.
Usage: EVEMT_GRID=<grid> python src/c1/td_tokens.py -> results/c1_td_tokens<TAG>.parquet"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, F0, T0, ROOT, TAG, terminal_cols, add_noise  # noqa: E402
from local_features import N, WLEN, LOOPS, line_params  # noqa: E402

W0 = 2 * np.pi * F0
PH = {"a": 0, "b": 1, "c": 2}


def der(x):
    return savgol_filter(x, 11, 2, deriv=1, delta=1 / FS)


def one(args):
    row, LP = args
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    sid = int(row["general/sim_idx"])
    f = D / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs = terminal_cols(hdr, m.group(1), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs)
    _x = add_noise([d[c].to_numpy(float) for c in cs], sid)
    i, v = _x[:3], _x[3:]
    Z1, Z0 = LP[line]
    R1, L1, R0, L0 = Z1.real, Z1.imag / W0, Z0.real, Z0.imag / W0
    i0 = (i[0] + i[1] + i[2]) / 3
    di = [der(x) for x in i]
    di0 = der(i0)
    loops = {}
    for lp in LOOPS:
        if lp.endswith("g"):
            p = PH[lp[0]]
            loops[lp] = (v[p], R1 * i[p] + L1 * di[p] + (R0 - R1) * i0 + (L0 - L1) * di0, i[p])
        else:
            p, q = PH[lp[0]], PH[lp[1]]
            loops[lp] = (v[p] - v[q], R1 * (i[p] - i[q]) + L1 * (di[p] - di[q]), i[p] - i[q])
    nf = int(round((row["events/event_start"] - T0) * FS))
    inc = "incipient" in row["events/event_type"]
    out = []
    for tau in (range(5, 51, 5) if inc else range(5, 81, 5)):
        s1 = nf + int(round(tau / 1000 * FS))
        a, b = s1 - WLEN + N, s1
        rec = dict(sim_idx=sid, tau_ms=tau)
        for lp, (vl, ul, il) in loops.items():
            vv, uu = vl[a:b], ul[a:b]
            gg = il[a:b] - il[a - N:b - N]
            X = np.stack([uu, gg], 1)
            beta, *_ = np.linalg.lstsq(X, vv, rcond=None)
            res = np.linalg.norm(vv - X @ beta) / (np.linalg.norm(vv) + 1e-12)
            drl = float(np.dot(vv, uu) / (np.dot(uu, uu) + 1e-12))
            rec[f"{lp}_dtd"] = float(np.clip(beta[0], -1, 2))
            rec[f"{lp}_drl"] = float(np.clip(drl, -1, 2))
            rec[f"{lp}_tdres"] = float(np.clip(res, 0, 5))
        out.append(rec)
    return out


def main():
    LP = line_params()
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    with ProcessPoolExecutor(12) as ex:
        res = [r for ch in ex.map(one, [(r, LP) for _, r in fl.iterrows()], chunksize=8) for r in ch]
    df = pd.DataFrame(res)
    out = ROOT / "results" / f"c1_td_tokens{TAG}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} windows, {df.shape[1] - 2} TD tokens -> {out}")


if __name__ == "__main__":
    main()
