"""Two-ended time-domain aerial-mode R-L locator (td_noise.py, SG derivative, clean) on adapt_grid-DoubleLine
line faults, windows by the reconstructed rule of design section 17a. No learning: R1, L1 from the adapt grid
model nameplate (graph_grid0.pickle; equal to the two-ended identified values to 4 digits on the benchmark grid).
Reports the TEST split (comparable to the in-grid tables) and all extracted episodes.
Output: results/c1_adapt_td.csv."""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import terminal_cols  # noqa: E402
from td_locator import clarke  # noqa: E402
from adapt_tokens import AD, SPLITS, line_params, window_ends  # noqa: E402

FS, W0, WLEN = 9600.0, 2 * np.pi * 50.0, 480


def der(x):
    return savgol_filter(x, 11, 2, deriv=1, delta=1 / FS)


def one(args):
    row, LP, split = args
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    sid = int(row["general/sim_idx"])
    f = AD / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    X = {k: d[k].to_numpy(float) for k in cs + cr}
    Z1 = LP[line][0]
    R, L = Z1.real, Z1.imag / W0
    iS, vS = clarke(*[X[c] for c in cs[:3]]), clarke(*[X[c] for c in cs[3:]])
    iR, vR = clarke(*[X[c] for c in cr[:3]]), clarke(*[X[c] for c in cr[3:]])
    modes = []
    for mi in range(2):
        a = vS[mi] - vR[mi] + R * iR[mi] + L * der(iR[mi])
        b = R * (iS[mi] + iR[mi]) + L * (der(iS[mi]) + der(iR[mi]))
        modes.append((a, b))
    nf, ends = window_ends(row, len(X[cs[0]]))
    out = []
    for s1 in ends:
        s0 = s1 - WLEN
        num = sum(float(np.dot(a[s0:s1], b[s0:s1])) for a, b in modes)
        den = sum(float(np.dot(b[s0:s1], b[s0:s1])) for a, b in modes) + 1e-12
        out.append(dict(sim_idx=sid, split=split, etype=row["events/event_type"], post_ms=(s1 - nf) / FS * 1000,
                        R=row["events/event_flt_shc_resistance"], loc_true=row["events/event_flt_target_line_location"],
                        loc_est=100 * num / den))
    return out


def main():
    LP = line_params()
    s = pd.read_csv(AD / "labels" / "settings_clean.csv")
    jobs = []
    for split in ("train", "test"):
        ids = {int(l.strip().split(":")[2]) for l in open(SPLITS / f"{split}.txt") if l.strip()}
        fl = s[s["general/sim_idx"].isin(ids) & s["events/event_type"].str.startswith("flt_")
               & s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()]
        jobs += [(r, LP, split) for _, r in fl.iterrows()]
    with ProcessPoolExecutor(12) as ex:
        df = pd.DataFrame([r for ch in ex.map(one, jobs, chunksize=8) for r in ch])
    df["err"] = (df.loc_est.clip(0, 100) - df.loc_true).abs()
    df.to_csv(Path(__file__).resolve().parents[2] / "results" / "c1_adapt_td.csv", index=False)
    for nm, g in (("test", df[df.split == "test"]), ("all", df)):
        shc = ~g.etype.str.contains("incipient")
        print(f"{nm}: n {len(g)}  MAE {g.err.mean():.3f}  median {g.err.median():.3f}  post<=15 {g.err[g.post_ms <= 15].mean():.3f}"
              f"  post>=21 {g.err[g.post_ms >= 21].mean():.3f}  SC-only {g.err[shc].mean():.3f}")
        print(g.groupby("etype").err.mean().round(3).to_string())


if __name__ == "__main__":
    main()
