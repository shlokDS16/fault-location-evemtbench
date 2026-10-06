"""C1 Stage B (first pass): physics locator on reconstructed benchmark windows (line view, DoubleLine
benchmark family). NOT yet digest-verified against the producer windows; labelled as a reconstruction.

Windows: start t_s = 1.0 + 0.005 k, length 50 ms (480 samples), fully inside the 1.0-1.5 s episode.
Fault-active: window overlaps the fault interval [t_f, t_f + duration] (duration = episode end for
SHC faults, 5 ms for incipient faults). tau = window_end - t_f (post-fault time inside the window).
Per-line impedance Z: median of per-episode pre-fault identifications of that line (normal-operation
calibration; parameter-free), taken from results/c1_stage_a_skip1p0.csv.
Locator per window:
  tau >= 21 ms : one-cycle DFT on the LAST cycle of the window, two-ended positive sequence with Z.
  3 ms <= tau  : least-squares sinusoid + offset + drift on post-fault samples in the window.
  tau < 3 ms   : prior = 50 % (line midpoint).
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, F0, T0, ROOT, TAG, phasor, pos_seq, terminal_cols  # noqa: E402
from sweep_tau import ls_phasor  # noqa: E402

WLEN, STEP = 480, 48
ZLINE = None


def zline():
    d = pd.read_csv(ROOT / "results" / f"c1_stage_a_skip1p0{TAG}.csv")
    return {k: complex(g.Z_re.median(), g.Z_im.median()) for k, g in d.groupby("line")}


def one(args):
    row, Zmap = args
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    f = D / "data" / f"result{int(row['general/sim_idx'])}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    X = {k: d[k].to_numpy(float) for k in cs + cr}
    n_tot = len(d)
    N = int(round(FS / F0))
    nf = int(round((row["events/event_start"] - T0) * FS))
    dur = 0.005 if "incipient" in row["events/event_type"] else 10.0
    nf_end = nf + int(round(dur * FS))
    Z = Zmap[line]
    out = []
    for k in range(0, (n_tot - WLEN) // STEP + 1):
        s0, s1 = k * STEP, k * STEP + WLEN
        if s1 <= nf or s0 >= nf_end:
            continue  # not fault-active
        tau_ms = (s1 - nf) / FS * 1000
        if tau_ms >= 21:
            st = s1 - N

            def sq(cols):
                return (pos_seq(*(phasor(X[c], st, N) for c in cols[3:])),
                        pos_seq(*(phasor(X[c], st, N) for c in cols[:3])))
        elif tau_ms >= 3:
            i0 = max(s0, nf + int(round(0.001 * FS)))

            def sq(cols):
                return (pos_seq(*(ls_phasor(X[c], i0, s1) for c in cols[3:])),
                        pos_seq(*(ls_phasor(X[c], i0, s1) for c in cols[:3])))
        else:
            sq = None
        if sq is None:
            est = 50.0
        else:
            VS, IS = sq(cs); VR, IR = sq(cr)
            est = 100 * float(np.real((VS - VR + Z * IR) / (Z * (IS + IR))))
        out.append(dict(sim_idx=int(row["general/sim_idx"]), etype=row["events/event_type"], k=k, tau_ms=tau_ms,
                        loc_true=row["events/event_flt_target_line_location"], loc_est=est))
    return out


def main():
    Zmap = zline()
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    with ProcessPoolExecutor(12) as ex:
        res = [r for ch in ex.map(one, [(r, Zmap) for _, r in fl.iterrows()], chunksize=8) for r in ch]
    df = pd.DataFrame(res)
    df["err"] = (df.loc_est.clip(0, 100) - df.loc_true).abs()
    df.to_csv(ROOT / "results" / f"c1_window_eval{TAG}.csv", index=False)
    df["tau_bin"] = pd.cut(df.tau_ms, [0, 3, 5, 7.5, 10, 15, 21, 50, 1000], right=False)
    print(f"windows={len(df)} episodes={df.sim_idx.nunique()}  MAE={df.err.mean():.3f}  median={df.err.median():.3f}")
    print(df.groupby("tau_bin", observed=True).err.agg(["count", "mean", "median"]).round(3).to_string())
    print(df.groupby("etype").err.agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
